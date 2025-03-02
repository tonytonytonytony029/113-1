#!/usr/bin/env python3
# coding: utf-8
# dual_path_routing.py

from ryu.base import app_manager
from ryu.controller import ofp_event, dpset
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
import networkx as nx
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response
import json

instance_name = 'dual_path_api_app'

class DualPathRouting(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(DualPathRouting, self).__init__(*args, **kwargs)
        wsgi = kwargs['wsgi']
        wsgi.register(DualPathRestController, {instance_name: self})

        self.net = nx.DiGraph()
        self.datapaths = {}
        self.hosts = {}  # mac: (dpid, port_no)
        self.last_nodes = set()
        self.last_edges = set()
        self.spanning_tree = None
        self.root_dpid = None
        self.installed_groups = {}  # (src_mac, dst_mac): group_id
        self.group_id_counter = 1
        self.installed_flows = set()  # 已安裝的流表記錄

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofproto = dp.ofproto

        # 安裝 table-miss 流表項
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(dp, priority=0, match=match, actions=actions)

    def add_flow(self, dp, priority, match, actions, group_id=None, idle_timeout=0, hard_timeout=0):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser

        if group_id:
            actions = [parser.OFPActionGroup(group_id)]

        # 使用 JSON 字符串確保 match 可哈希
        match_json = json.dumps(match.to_jsondict(), sort_keys=True)
        actions_tuple = tuple((action.__class__.__name__, getattr(action, 'port', None)) for action in actions)
        flow_id = (dp.id, priority, match_json, actions_tuple)

        if flow_id in self.installed_flows:
            return
        self.installed_flows.add(flow_id)

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=priority,
                                match=match, instructions=inst,
                                idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        dp.send_msg(mod)
        self.logger.debug("安裝流表: DPID=%s, Priority=%s, Match=%s, Actions=%s",
                          dp.id, priority, match_json, actions)

    def update_topology(self):
        switches = get_switch(self, None)
        links = get_link(self, None)

        self.net.clear()
        current_nodes = set()
        current_edges = set()

        for sw in switches:
            dpid = sw.dp.id
            self.net.add_node(dpid)
            self.datapaths[dpid] = sw.dp
            current_nodes.add(dpid)

        for l in links:
            self.net.add_edge(l.src.dpid, l.dst.dpid, port=l.src.port_no)
            self.net.add_edge(l.dst.dpid, l.src.dpid, port=l.dst.port_no)
            current_edges.add((l.src.dpid, l.dst.dpid))
            current_edges.add((l.dst.dpid, l.src.dpid))

        for mac, (dpid, port_no) in self.hosts.items():
            self.net.add_node(mac)
            self.net.add_edge(dpid, mac, port=port_no)
            self.net.add_edge(mac, dpid)
            current_nodes.add(mac)
            current_edges.add((dpid, mac))
            current_edges.add((mac, dpid))

        if switches:
            self.root_dpid = min([sw.dp.id for sw in switches])
            undirected = self.net.to_undirected()
            if self.root_dpid in undirected.nodes:
                self.spanning_tree = nx.bfs_tree(undirected, self.root_dpid)

        if current_nodes != self.last_nodes or current_edges != self.last_edges:
            self.logger.info("當前節點: %s", current_nodes)
            self.logger.info("當前邊: %s", current_edges)
            self.last_nodes = current_nodes
            self.last_edges = current_edges

    @set_ev_cls([event.EventSwitchEnter, event.EventSwitchLeave, event.EventLinkAdd, event.EventLinkDelete], MAIN_DISPATCHER)
    def topology_change_handler(self, ev):
        self.update_topology()

    @set_ev_cls(event.EventHostAdd, MAIN_DISPATCHER)
    def host_add_handler(self, ev):
        host = ev.host
        mac = host.mac
        dpid = host.port.dpid
        port_no = host.port.port_no

        self.logger.info("處理主機加入事件: MAC=%s, DPID=%s, Port=%s", mac, dpid, port_no)

        if mac.startswith("00:00:00:00:00:"):
            self.hosts[mac] = (dpid, port_no)
            self.update_topology()
            self.logger.info("主機新增: MAC=%s 連接到交換機=%s 端口=%s", mac, dpid, port_no)
        else:
            self.logger.info("忽略非 Mininet 主機 MAC: %s", mac)

    @set_ev_cls(dpset.EventDP, [dpset.DPSET_EV_DISPATCHER])
    def datapath_state_change_handler(self, ev):
        dp = ev.dp
        if ev.enter:
            self.datapaths[dp.id] = dp
        else:
            if dp.id in self.datapaths:
                del self.datapaths[dp.id]

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        parser = dp.ofproto_parser
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        in_port = msg.match['in_port']

        if eth.ethertype == 0x88cc:
            return

        src_mac = eth.src
        dst_mac = eth.dst

        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            self.handle_arp(dp, in_port, pkt, arp_pkt, msg.data)
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            self.handle_ipv4(dp, in_port, pkt, ip_pkt, src_mac, dst_mac, msg)

    def handle_arp(self, dp, in_port, pkt, arp_pkt, data):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        src_ip = arp_pkt.src_ip
        dst_ip = arp_pkt.dst_ip
        src_mac = arp_pkt.src_mac
        dst_mac = self.ip_to_mac(dst_ip)

        self.logger.debug("處理 ARP 封包: Opcode=%s, Src_IP=%s, Dst_IP=%s, Src_MAC=%s, Dst_MAC=%s",
                          arp_pkt.opcode, src_ip, dst_ip, src_mac, dst_mac)

        if arp_pkt.opcode == arp.ARP_REQUEST:
            self.logger.info("收到 ARP 請求: %s 詢問 %s", src_mac, dst_ip)
            if dst_mac and dst_mac in self.hosts:
                self.logger.info("已知目標 MAC %s: %s", dst_ip, dst_mac)
                self.send_arp_reply(dp, arp_pkt, dst_mac)
            else:
                self.logger.info("未知目標 MAC %s，沿生成樹廣播 ARP", dst_ip)
                self.flood_arp(dp, data)
        elif arp_pkt.opcode == arp.ARP_REPLY:
            self.logger.info("收到 ARP 回覆: %s 是 %s", src_ip, src_mac)
            # 不需額外處理，host_add已記錄主機位置

    def handle_ipv4(self, dp, in_port, pkt, ip_pkt, src_mac, dst_mac, msg):
        src_ip = ip_pkt.src
        dst_ip = ip_pkt.dst

        self.logger.info("處理 IPv4 封包: %s (%s) -> %s (%s)", src_mac, src_ip, dst_mac, dst_ip)

        if src_mac not in self.hosts or dst_mac not in self.hosts:
            self.logger.info("未知的源或目的 MAC，忽略封包")
            return

        try:
            path1, path2 = self.compute_two_disjoint_paths(src_mac, dst_mac)
            self.logger.info("找到兩條分離路徑 %s 和 %s", path1, path2)
        except ValueError as e:
            self.logger.warning("無法計算兩條分離路徑: %s，嘗試使用單一路徑作為回退", e)
            try:
                path1 = nx.shortest_path(self.net, src_mac, dst_mac)
                path2 = None
                self.logger.info("使用單一路徑 %s 作為回退", path1)
            except nx.NetworkXNoPath:
                self.logger.error("無法在 %s 和 %s 之間找到任何路徑", src_mac, dst_mac)
                return

        parser = dp.ofproto_parser
        if path2:
            group_id = self.get_group_id(src_mac, dst_mac)
            self.install_group(dp, group_id, path1, path2)
            match = parser.OFPMatch(eth_type=0x0800, eth_src=src_mac, eth_dst=dst_mac)
            actions = [parser.OFPActionGroup(group_id)]
            self.add_flow(dp, priority=20, match=match, actions=actions)
        else:
            self.logger.info("只使用單一路徑安裝流表")
            self.install_single_path(dp, path1, src_mac, dst_mac)
            out_port = self.get_out_port(path1, src_mac, dst_mac)
            if out_port:
                actions = [parser.OFPActionOutput(out_port)]
                out = parser.OFPPacketOut(
                    datapath=dp,
                    buffer_id=msg.buffer_id,
                    in_port=in_port,
                    actions=actions,
                    data=msg.data
                )
                dp.send_msg(out)
            else:
                self.logger.warning("無法找到出端口，封包轉發失敗")

    def send_arp_reply(self, dp, arp_pkt, dst_mac):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        src_ip = arp_pkt.dst_ip
        dst_ip = arp_pkt.src_ip
        src_mac = dst_mac
        dst_mac = arp_pkt.src_mac

        e = ethernet.ethernet(dst=dst_mac, src=src_mac, ethertype=0x0806)
        a = arp.arp(opcode=arp.ARP_REPLY,
                    src_mac=src_mac, src_ip=src_ip,
                    dst_mac=dst_mac, dst_ip=dst_ip)
        p = packet.Packet()
        p.add_protocol(e)
        p.add_protocol(a)
        p.serialize()
        arp_reply = p.data

        (h_dpid, h_port) = self.hosts[dst_mac]
        out_port = h_port

        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=arp_reply
        )
        dp.send_msg(out)
        self.logger.info("發送 ARP 回覆: %s -> %s (DPID=%s, Port=%s)",
                         src_mac, dst_mac, h_dpid, h_port)

    def flood_arp(self, dp, data):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser

        if self.spanning_tree and self.root_dpid in self.spanning_tree:
            undirected = self.net.to_undirected()
            if dp.id in undirected:
                descendants = nx.descendants(self.spanning_tree, dp.id)
                for neighbor in self.net[dp.id]:
                    if neighbor in descendants and isinstance(neighbor, int):
                        out_port = self.net[dp.id][neighbor]['port']
                        actions = [parser.OFPActionOutput(out_port)]
                        out = parser.OFPPacketOut(
                            datapath=dp,
                            buffer_id=ofproto.OFP_NO_BUFFER,
                            in_port=ofproto.OFPP_CONTROLLER,
                            actions=actions,
                            data=data
                        )
                        dp.send_msg(out)
                        self.logger.debug("ARP Flood 至 DPID=%s, Port=%s", neighbor, out_port)
        else:
            self.logger.warning("尚無生成樹，廣播 ARP 至所有交換機")
            for neighbor in self.net[dp.id]:
                if isinstance(neighbor, int):
                    out_port = self.net[dp.id][neighbor]['port']
                    actions = [parser.OFPActionOutput(out_port)]
                    out = parser.OFPPacketOut(
                        datapath=dp,
                        buffer_id=ofproto.OFP_NO_BUFFER,
                        in_port=ofproto.OFPP_CONTROLLER,
                        actions=actions,
                        data=data
                    )
                    dp.send_msg(out)
                    self.logger.debug("ARP Flood 至 DPID=%s, Port=%s", neighbor, out_port)

    def ip_to_mac(self, ip):
        if ip.startswith('10.0.0.'):
            try:
                host_id = int(ip.split('.')[-1])
                mac = "00:00:00:00:00:%02x" % host_id
                if mac in self.hosts:
                    return mac
            except ValueError:
                pass
        return None

    def install_group(self, dp, group_id, path1, path2):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser

        if len(path1) < 2 or len(path2) < 2:
            self.logger.error("路徑長度不足，無法安裝群組表")
            return

        n1 = path1[0]
        n2 = path1[1]
        out_port1 = self.net[n1][n2]['port']
        actions1 = [parser.OFPActionOutput(out_port1)]

        n1 = path2[0]
        n2 = path2[1]
        out_port2 = self.net[n1][n2]['port']
        actions2 = [parser.OFPActionOutput(out_port2)]

        buckets = [
            parser.OFPBucket(actions=actions1),
            parser.OFPBucket(actions=actions2)
        ]

        group_mod = parser.OFPGroupMod(
            datapath=dp,
            command=ofproto.OFPGC_ADD,
            type=ofproto.OFPGT_SELECT,
            group_id=group_id,
            buckets=buckets
        )
        dp.send_msg(group_mod)
        self.logger.info("安裝群組 %s，路徑1出端口 %s，路徑2出端口 %s", group_id, out_port1, out_port2)

    def install_single_path(self, dp, path, src_mac, dst_mac, priority=20):
        for i in range(len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            if n1 in self.datapaths and self.net.has_edge(n1, n2):
                dp_current = self.datapaths[n1]
                parser_current = dp_current.ofproto_parser
                out_port = self.net[n1][n2]['port']
                match = parser_current.OFPMatch(eth_type=0x0800, eth_src=src_mac, eth_dst=dst_mac)
                actions = [parser_current.OFPActionOutput(out_port)]
                self.add_flow(dp_current, priority, match, actions)
                self.logger.debug("安裝單一路徑流表: DPID=%s, Match=%s, Actions=%s", dp_current.id, match, actions)

    def get_group_id(self, src_mac, dst_mac):
        key = (src_mac, dst_mac)
        if key not in self.installed_groups:
            group_id = self.group_id_counter
            self.installed_groups[key] = group_id
            self.group_id_counter += 1
        return self.installed_groups[key]

    def compute_two_disjoint_paths(self, src_mac, dst_mac):
        if src_mac not in self.net or dst_mac not in self.net:
            raise ValueError("源或目的節點不在網路圖中")

        try:
            paths = list(nx.shortest_simple_paths(self.net, src_mac, dst_mac))
        except nx.NetworkXNoPath:
            raise ValueError("無法在 {} 和 {} 之間找到路徑".format(src_mac, dst_mac))

        if not paths:
            raise ValueError("無法在 {} 和 {} 之間找到任何路徑".format(src_mac, dst_mac))

        for i, path_a in enumerate(paths):
            for path_b in paths[i + 1:]:
                edges_a = set(zip(path_a, path_a[1:]))
                edges_b = set(zip(path_b, path_b[1:]))
                if edges_a.isdisjoint(edges_b):
                    return path_a, path_b

        path1 = paths[0]
        return path1, None

    def get_out_port(self, path, src, dst):
        for i in range(len(path)-1):
            n1 = path[i]
            n2 = path[i+1]
            if n1 in self.datapaths:
                return self.net[n1][n2]['port']
        return None

class DualPathRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(DualPathRestController, self).__init__(req, link, data, **config)
        self.dp_app = data[instance_name]

    @route('dualpath', '/dualpath/hosts', methods=['GET'])
    def list_hosts(self, req, **kwargs):
        hosts_info = {mac: {"dpid": dpid, "port": port} for mac, (dpid, port) in self.dp_app.hosts.items()}
        return Response(content_type='application/json', body=json.dumps(hosts_info))

    @route('dualpath', '/dualpath/route/{src_mac}/{dst_mac}', methods=['GET'])
    def get_dual_path(self, req, **kwargs):
        src_mac = kwargs['src_mac']
        dst_mac = kwargs['dst_mac']
        if src_mac in self.dp_app.hosts and dst_mac in self.dp_app.hosts:
            try:
                path1, path2 = self.dp_app.compute_two_disjoint_paths(src_mac, dst_mac)
                body = {'path1': path1, 'path2': path2 if path2 else '無可用的第二條路徑'}
                return Response(content_type='application/json', body=json.dumps(body))
            except ValueError as e:
                return Response(status=404, body=str(e))
        else:
            return Response(status=404, body='未知的主機。')
