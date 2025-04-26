from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

PII_CATEGORIES = {
    "Customer Name PII": ["customer_name", "full_name", "first_name", "last_name", "John Doe", "Jane Muller"],
    "Email Address PII": ["email", "email address", "user@example.com", "jane.doe@gmail.com"],
    "Phone Number PII": ["phone", "phone number", "555-1234", "+1-800-555-1212"],
    "Government ID PII": ["ssn", "social security number", "123-45-6789", "passport number", "national id"],
    "Date of Birth PII": ["date of birth", "dob", "1990-01-01"],
    "Bank Account PII": ["iban", "account number", "bank account", "credit card number", "1234567812345678"],
    "Address PII": ["address", "home address", "123 Main St", "zipcode"],
}

pii_examples = [example for examples in PII_CATEGORIES.values() for example in examples]
example_to_label = {example: label for label, examples in PII_CATEGORIES.items() for example in examples}
example_embeddings = model.encode(pii_examples, convert_to_tensor=True)

def get_best_label(texts):
    if not texts:
        return 0.0, None

    embeddings = model.encode(texts, convert_to_tensor=True)
    sims = util.cos_sim(embeddings, example_embeddings)

    best_score = 0.0
    best_label = None

    for sim_vector in sims:
        max_idx = sim_vector.argmax().item()
        max_score = sim_vector[max_idx].item()
        label = example_to_label[pii_examples[max_idx]]

        if max_score > best_score:
            best_score = max_score
            best_label = label

    return best_score, best_label

def predict_pii(columns, name_score_threshold=0.6, sample_score_threshold=0.6):
    results = []

    for col in columns:
        name_score, name_label = get_best_label([col["name"]])
        sample_score, sample_label = get_best_label(col.get("samples", []))

        # Final decision
        if name_score >= name_score_threshold and sample_score >= sample_score_threshold:
            final_label = name_label if name_label == sample_label else f"{name_label or ''} / {sample_label or ''}".strip(" /")
            pii_score = max(name_score, sample_score)
        else:
            final_label = "Non-PII"
            pii_score = max(name_score, sample_score)

        results.append({
            "name": col["name"],
            "type": col["type"],
            "name_score": round(name_score, 3),
            "sample_score": round(sample_score, 3),
            "pii_score": round(pii_score, 3),
            "pii_type": final_label
        })

    return results
