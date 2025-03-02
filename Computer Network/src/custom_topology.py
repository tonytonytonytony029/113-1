from mininet.topo import Topo

class CustomTopo(Topo):
    def build(self):

        switches = {}
        for i in range(1, 9):
            switches['S{}'.format(i)] = self.addSwitch('s{}'.format(i))

        hosts = {}
        for i in range(1, 10):
            hosts['H{}'.format(i)] = self.addHost('h{}'.format(i), mac="00:00:00:00:00:{:02x}".format(i))
        self.addLink(hosts['H1'], switches['S1'])
        self.addLink(hosts['H2'], switches['S3'])
        self.addLink(hosts['H3'], switches['S6'])
        self.addLink(hosts['H4'], switches['S5'])
        self.addLink(hosts['H5'], switches['S5'])
        self.addLink(hosts['H6'], switches['S8'])
        self.addLink(hosts['H7'], switches['S8'])
        self.addLink(hosts['H8'], switches['S6'])
        self.addLink(hosts['H9'], switches['S4'])

        links = [
            ('S1', 'S2'), ('S1', 'S3'), ('S2', 'S3'), ('S2', 'S5'),
            ('S2', 'S6'), ('S3', 'S4'), ('S4', 'S5'), ('S4', 'S8'),
            ('S5', 'S7'), ('S5', 'S8'), ('S6', 'S7'), ('S7', 'S8')
        ]
        for src, dst in links:
            self.addLink(switches[src], switches[dst])

topos = {'customtopo': CustomTopo}
