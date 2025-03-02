# %%
import json, os, re, logging, csv, torch, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup


# %%
MAIN_DATA_PATHS = {
    'train': 'train.json',
    'valid': 'valid.json',
    'test': 'test.json'
}
ARTICLE_FOLDER = os.path.abspath('C:/Users/林滋隆/OneDrive - 國立陽明交通大學/桌面/IRhw2/articles')
OUTPUT_FILE_PATHS = {
    'train': 'train_processed.json',
    'valid': 'valid_processed.json',
    'test': 'test_processed.json'
}
OUTPUT_CSV_PATH = 'predictions.csv'
MODEL_SAVE_PATH = 'fine_tuned_model'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# %%
def clean_text(text):
    return re.compile(r'<[^>]+>').sub('', text).lower()

def extract_evidence(claim_text, sentences, fixed_threshold=0.38):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([claim_text] + sentences)
    similarities = (vectors[1:] * vectors[0].T).toarray().flatten()
    evidence_sentences, seen = [], set()
    for i, sim in enumerate(similarities):
        if sim > fixed_threshold:
            sentence = sentences[i].strip()
            if sentence and sentence not in seen:
                evidence_sentences.append(sentence)
                seen.add(sentence)
    logger.debug(f"Evidences extracted: {evidence_sentences}")
    return evidence_sentences

def process_single_claim(claim, article_folder):
    claim_id = claim.get('metadata', {}).get('id', 'unknown')
    claim_text = clean_text(claim.get('metadata', {}).get('claim', ''))
    articles = claim.get('metadata', {}).get('premise_articles', {})
    evidences = []
    for _, article_file in articles.items():
        article_path = os.path.join(article_folder, article_file)
        try:
            with open(article_path, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
            logger.info(f"成功載入文章: {article_file}")
        except Exception as e:
            logger.error(f"載入文章失敗 {article_file}: {e}")
            continue
        if isinstance(article_data, list):
            content = ' '.join(article_data)
        elif isinstance(article_data, dict):
            content = article_data.get('content', '')
        else:
            logger.warning(f"未知的文章格式: {article_file}")
            content = ''
        sentences = [s.strip() for s in clean_text(content).split('.') if s.strip()]
        evidences.extend(extract_evidence(claim_text, sentences))
    return {"id": claim_id, "claim": claim_text, "evidences": evidences}


# %%
def process_data(main_data_path, article_folder, output_file_path):
    try:
        with open(main_data_path, 'r', encoding='utf-8') as f:
            main_data = json.load(f)
        logger.info(f"成功載入主資料: {main_data_path}")
    except Exception as e:
        logger.critical(f"主資料載入失敗: {e}")
        return
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_claim = {executor.submit(process_single_claim, claim, article_folder): claim for claim in main_data}
        for future in tqdm(as_completed(future_to_claim), total=len(future_to_claim), desc="Processing Claims"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing claim: {e}")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"結果已保存至 {output_file_path}")
    except Exception as e:
        logger.error(f"寫入結果失敗: {e}")


# %%
for split, path in MAIN_DATA_PATHS.items():
    output_path = OUTPUT_FILE_PATHS.get(split)
    if output_path:
        process_data(path, ARTICLE_FOLDER, output_path)


# %%
class Args:
    mode = 'both'
    model_path = 'saved_model'
    train_data = 'train_processed.json'
    valid_data = 'valid_processed.json'
    test_data = 'test_processed.json'
    output_csv = OUTPUT_CSV_PATH
    batch_size = 8
    num_epochs = 3
    learning_rate = 1e-5
    max_length = 256
    weight_decay = 0.01
    warmup_ratio = 0.1
    num_workers = 4

args = Args()


# %%
class ClaimDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.labels = []
        for entry in data:
            claim = entry.get("claim", "")
            evidences = ' '.join(entry.get("evidences", []))
            input_text = f"{claim} </s></s> {evidences}"
            inputs = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.inputs.append(inputs)
            self.labels.append(entry.get("label", 0))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        inputs = {k: v.squeeze(0) for k, v in self.inputs[idx].items()}
        label = torch.tensor(self.labels[idx])
        return inputs, label

class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.claim_ids = []
        for entry in data:
            claim_id = entry.get("id", f"unknown_id_{len(self.claim_ids)}")
            claim = entry.get("claim", "")
            evidences = ' '.join(entry.get("evidences", []))
            input_text = f"{claim} </s></s> {evidences}"
            inputs = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.inputs.append(inputs)
            self.claim_ids.append(claim_id)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        inputs = {k: v.squeeze(0) for k, v in self.inputs[idx].items()}
        claim_id = self.claim_ids[idx]
        return inputs, claim_id


# %%
def train(model, tokenizer, train_loader, val_loader, device, args):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Average Training Loss: {avg_train_loss:.4f}")
        model.eval()
        val_labels = []
        val_predictions = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                for i, (pred, label) in enumerate(zip(predictions, labels)):
                    if pred != label:
                        claim_text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
                        print(f"Claim ID: {i}")
                        print(f"Claim Text: {claim_text}")
                        print(f"True Label: {label.item()}, Predicted Label: {pred.item()}")
        val_f1 = f1_score(val_labels, val_predictions, average='macro')
        print(f"Epoch {epoch+1}/{args.num_epochs} - Validation Macro F1 Score: {val_f1:.4f}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model and tokenizer saved to '{MODEL_SAVE_PATH}' directory.")


# %%
def predict(model, tokenizer, test_loader, device, output_csv):
    model.eval()
    csv_data = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, claim_ids = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            for claim_id, prediction in zip(claim_ids, predictions):
                csv_data.append({"id": claim_id, "rating": prediction})
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["id", "rating"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"CSV file '{output_csv}' generated successfully with {len(csv_data)} entries.")


# %%
def run_training_and_prediction(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if args.mode in ['train', 'both']:
        tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
        model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-large', num_labels=3).to(device)
    else:
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model path '{args.model_path}' does not exist. Cannot perform prediction.")
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_path)
        model = DebertaV2ForSequenceClassification.from_pretrained(args.model_path).to(device)
    if args.mode in ['train', 'both']:
        try:
            with open(args.train_data, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
        except Exception as e:
            logger.error(f"載入訓練資料失敗: {e}")
            train_data = []
        try:
            with open(args.valid_data, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
        except Exception as e:
            logger.error(f"載入驗證資料失敗: {e}")
            val_data = []
        train_dataset = ClaimDataset(train_data, tokenizer, max_length=args.max_length)
        val_dataset = ClaimDataset(val_data, tokenizer, max_length=args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train(model, tokenizer, train_loader, val_loader, device, args)
    if args.mode in ['predict', 'both']:
        try:
            with open(args.test_data, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except Exception as e:
            logger.error(f"載入測試資料失敗: {e}")
            test_data = []
        test_dataset = TestDataset(test_data, tokenizer, max_length=args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        predict(model, tokenizer, test_loader, device, args.output_csv)


# %%
run_training_and_prediction(args)



