import torch
import pickle
import json
import os
import pandas as pd
import re
from datetime import datetime
from collections import defaultdict
from model import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# COMPREHENSIVE VOCABULARY LISTS
# ===========================================================================

subjects = [
    "I", "You", "He", "She", "We", "They",
    "My mom", "My dad", "My brother", "My sister", "My friend", "My teacher",
    "My grandma", "My grandpa", "My cousin", "My neighbor",
    "The boy", "The girl", "The child", "The baby", "The kid",
    "The man", "The woman", "The teacher", "The student",
    "The doctor", "The nurse", "The driver", "The farmer", "The cook",
    "The police", "The guard", "The coach", "The trainer",
    "My dog", "My cat", "My pet", "The dog", "The cat", "The bird",
    "A boy", "A girl", "A child", "A baby", "A kid",
    "One boy", "One girl", "Some kids", "Many people",
    "This boy", "That girl", "These kids", "Those people",
    "My classmate", "My best friend", "My school friend", "My play friend",
    "My uncle", "My aunty", "My mother", "My father",
    "Our teacher", "Our friend", "Our class", "Our group",
    "Your mom", "Your dad", "Your friend", "Your teacher",
    "His friend", "Her friend", "Their friend",
    "The small boy", "The small girl", "The big boy", "The big girl",
    "The little child", "The young boy", "The young girl",
    "The old man", "The old woman",
    "Someone", "Anybody", "Nobody", "Everyone",
    "The helper", "The shopkeeper", "The chef", "The worker",
    "The principal", "The player", "The singer", "The dancer",
    "The artist", "The reader", "The writer", "The learner"
]

places = [
    "park", "school", "home", "shop", "garden", "playground", "classroom",
    "kitchen", "bathroom", "bedroom", "hall", "yard", "street", "road",
    "market", "mall", "hospital", "clinic", "library", "office",
    "bus stop", "railway station", "airport",
    "temple", "church", "mosque",
    "restaurant", "hotel", "cafe", "bakery",
    "gym", "stadium", "field", "zoo", "museum", "cinema",
    "village", "city", "town", "house", "building", "apartment",
    "lab", "computer lab", "staff room",
    "play area", "kids zone", "swimming pool",
    "school gate", "main road", "side road",
    "friend house", "teacher room",
    "shopping area", "food court", "parking area",
    "tuition center", "music class", "dance class", "art class",
    "sports ground", "football field", "cricket ground",
    "city park", "local market", "big mall",
    "corner shop", "bus stand", "railway platform",
    "small shop", "big store", "game zone", "reading room", "study room",
    "play school", "daycare", "training center", "workplace", "office room"
]

objects = [
    "water", "toy", "food", "book", "ball", "apple", "banana", "mango", "orange",
    "grapes", "rice", "roti", "bread", "egg", "milk", "juice", "tea", "coffee",
    "sugar", "salt", "cake", "biscuit", "chocolate",
    "pen", "pencil", "eraser", "bag", "notebook", "paper", "box", "bottle",
    "doll", "car", "blocks", "puzzle", "game",
    "chair", "table", "bed", "sofa", "fan", "light", "door", "window", "clock",
    "phone", "laptop", "tablet", "tv", "remote",
    "shirt", "pants", "dress", "shoes", "socks", "cap",
    "plate", "spoon", "fork", "cup", "glass",
    "brush", "toothbrush", "soap", "towel", "comb",
    "lunch box", "water bottle",
    "key", "lock", "gift", "flower", "plant", "tree",
    "bus", "bike", "car", "train", "auto",
    "money", "ticket", "card",
    "homework", "project", "drawing", "story",
    "snack", "meal", "breakfast", "lunch", "dinner",
    "toy car", "school bag", "story book", "note", "file"
]

emotions = [
    "happy", "sad", "angry", "scared", "excited", "tired", "sleepy",
    "hungry", "thirsty", "bored", "confused", "surprised",
    "worried", "calm", "relaxed", "proud", "shy", "quiet",
    "friendly", "kind", "good", "bad", "okay", "fine",
    "sick", "hurt", "better", "not well",
    "ready", "busy", "free",
    "strong", "weak", "fast", "slow",
    "hot", "cold", "warm", "cool",
    "clean", "dirty", "wet", "dry",
    "full", "empty", "big", "small", "short", "tall",
    "new", "old", "young", "early", "late",
    "right", "wrong",
    "safe", "unsafe", "alone", "together",
    "inside", "outside", "near", "far",
    "easy", "hard", "simple", "difficult",
    "fun", "boring", "interesting",
    "loud", "soft", "bright", "dark",
    "colorful", "plain", "beautiful", "ugly"
]

# Action verbs with their natural usages
action_verbs = {
    "go": ["go to", "going to", "went to"],
    "play": ["play", "playing", "played"],
    "eat": ["eat", "eating", "ate"],
    "drink": ["drink", "drinking", "drank"],
    "sleep": ["sleep", "sleeping", "slept"],
    "read": ["read", "reading"],
    "write": ["write", "writing", "wrote"],
    "draw": ["draw", "drawing", "drew"],
    "watch": ["watch", "watching", "watched"],
    "walk": ["walk", "walking", "walked"],
    "run": ["run", "running", "ran"],
    "talk": ["talk", "talking", "talked"],
    "study": ["study", "studying", "studied"],
    "learn": ["learn", "learning", "learned"],
    "help": ["help", "helping", "helped"],
    "want": ["want", "wanting", "wanted"],
    "need": ["need", "needing", "needed"],
    "like": ["like", "liking", "liked"],
    "love": ["love", "loving", "loved"],
    "sit": ["sit", "sitting", "sat"],
    "stand": ["stand", "standing", "stood"],
    "listen": ["listen", "listening", "listened"],
    "look": ["look", "looking", "looked"],
    "see": ["see", "seeing", "saw"],
    "buy": ["buy", "buying", "bought"],
    "take": ["take", "taking", "took"],
    "give": ["give", "giving", "gave"],
    "open": ["open", "opening", "opened"],
    "close": ["close", "closing", "closed"],
    "find": ["find", "finding", "found"],
    "search": ["search", "searching", "searched"]
}

# ===========================================================================
# TYPO CORRECTION MAPPING
# ===========================================================================

typo_corrections = {
    # Play variations
    "plai": "play", "pla": "play", "pley": "play", "ply": "play",
    "plaay": "play", "playy": "play", "pai": "play",
    
    # Friend variations
    "freind": "friend", "frend": "friend", "freindd": "friend",
    "frnd": "friend", "frands": "friends", "frends": "friends",
    "freinds": "friends", "fiend": "friend", "fren": "friend",
    
    # School variations
    "skool": "school", "schol": "school", "scl": "school",
    "scool": "school", "shool": "school", "skhool": "school",
    
    # Drink variations
    "drnk": "drink", "dink": "drink", "drik": "drink",
    "drnik": "drink", "dirk": "drink",
    
    # Eat variations
    "et": "eat", "eet": "eat", "eaat": "eat", "aet": "eat",
    
    # Sleep variations
    "slep": "sleep", "slp": "sleep", "sleap": "sleep",
    "sleep": "sleep", "slepp": "sleep",
    
    # Want variations
    "wnat": "want", "wnt": "want", "waant": "want",
    "wantt": "want", "wan": "want",
    
    # Go variations
    "goo": "go", "gooo": "go", "ggo": "go",
    
    # Water variations
    "watr": "water", "wter": "water", "wateer": "water",
    "watter": "water",
    
    # Food variations
    "fod": "food", "fud": "food", "foood": "food",
    "foof": "food",
    
    # Juice variations
    "juse": "juice", "juce": "juice", "juise": "juice",
    "jucie": "juice",
    
    # Happy variations
    "hapy": "happy", "happi": "happy", "hapi": "happy",
    "happpy": "happy",
    
    # Sad variations
    "saad": "sad", "sadd": "sad", "sade": "sad",
    
    # Tired variations
    "tierd": "tired", "tird": "tired", "tyred": "tired",
    "tiredd": "tired",
    
    # Hungry variations
    "hungri": "hungry", "hungury": "hungry", "hungrey": "hungry",
    "hungy": "hungry",
    
    # Home variations
    "hom": "home", "hoome": "home", "homee": "home",
    
    # Park variations
    "parc": "park", "parkk": "park", "prk": "park",
    
    # Book variations
    "bok": "book", "boook": "book", "buk": "book",
    
    # Help variations
    "halp": "help", "hlp": "help", "healp": "help"
}

# ===========================================================================
# LOAD MODEL AND VOCABULARY
# ===========================================================================

with open("vocab.pkl", "rb") as f:
    word2idx = pickle.load(f)

idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)

# Initialize model
encoder = Encoder(vocab_size, 128, 256)
decoder = Decoder(vocab_size, 128, 256)
model = Seq2Seq(encoder, decoder, device).to(device)

# Load trained weights
model.load_state_dict(torch.load("asd_model.pt", map_location=device))
model.eval()

# Load test data
test_df = pd.read_csv("asd_test.csv")
test_data = list(zip(test_df["fragmented_input"], test_df["corrected_output"]))

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def numericalize(sentence):
    """Convert sentence to token indices"""
    return [word2idx.get(word, word2idx["<unk>"]) for word in sentence.lower().split()]


def clean_input(text):
    """Clean and normalize user input"""
    text = text.strip()
    text = text.split(",")[0]  # Remove everything after comma
    text = text.replace("?", "").replace("!", "").replace(".", "")
    return text.lower()


def apply_typo_corrections(text):
    """Apply typo corrections to input text"""
    words = text.split()
    corrected_words = []
    
    for word in words:
        if word in typo_corrections:
            corrected_words.append(typo_corrections[word])
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words)


def exact_word_match(text, word_list):
    """
    Extract words from word_list that appear as EXACT COMPLETE WORDS in text.
    Prevents partial matching inside other words.
    """
    text = f" {text} "  # Add spaces for boundary matching
    matched = []
    
    for word in word_list:
        word_lower = word.lower()
        # Check for exact word boundaries
        if f" {word_lower} " in text:
            matched.append(word_lower)
    
    return list(set(matched))  # Remove duplicates


def detect_entities(text):
    """
    Detect all entities (subjects, places, objects, emotions, actions) from input text.
    Uses exact word matching to avoid false positives.
    """
    entities = {
        "subjects": [],
        "places": [],
        "objects": [],
        "emotions": [],
        "actions": []
    }
    
    # Normalize text for matching
    text = f" {text} "
    
    # Match subjects (can be multi-word)
    for subject in subjects:
        subject_lower = subject.lower()
        if f" {subject_lower} " in text:
            entities["subjects"].append(subject_lower)
    
    # Match places
    entities["places"] = exact_word_match(text, places)
    
    # Match objects
    entities["objects"] = exact_word_match(text, objects)
    
    # Match emotions
    entities["emotions"] = exact_word_match(text, emotions)
    
    # Match actions
    for action in action_verbs.keys():
        if f" {action} " in text:
            entities["actions"].append(action)
    
    return entities


def infer_implicit_intents(entities, text):
    """
    Infer implicit intents based on context.
    E.g., "school friend" implies "go to school and play with friends"
    """
    inferred = []
    
    # If place + friend mentioned without explicit action
    if entities["places"] and ("friend" in text or "friends" in text):
        if "play" not in entities["actions"] and "go" not in entities["actions"]:
            inferred.append("implied_social_activity")
    
    # If emotion (tired/sleepy) + no action -> implies rest/sleep
    if any(e in ["tired", "sleepy"] for e in entities["emotions"]):
        if not entities["actions"]:
            inferred.append("implied_rest")
    
    # If emotion (hungry/thirsty) + no action -> implies eat/drink
    if "hungry" in entities["emotions"] and not entities["actions"]:
        inferred.append("implied_eat")
    
    if "thirsty" in entities["emotions"] and not entities["actions"]:
        inferred.append("implied_drink")
    
    # If object (food/drink related) + no action -> implies want/eat/drink
    food_objects = ["food", "apple", "banana", "rice", "bread", "cake", "chocolate"]
    drink_objects = ["water", "juice", "milk", "tea", "coffee"]
    
    if any(obj in entities["objects"] for obj in food_objects):
        if not entities["actions"]:
            inferred.append("implied_eat")
    
    if any(obj in entities["objects"] for obj in drink_objects):
        if not entities["actions"]:
            inferred.append("implied_drink")
    
    return inferred


def build_sentence_components(entities, text, inferred_intents):
    """
    Build sentence components from detected entities and inferred intents.
    Returns a list of action phrases.
    """
    components = []
    actions_used = set()
    
    # Handle explicit actions first
    for action in entities["actions"]:
        if action in actions_used:
            continue
        
        if action == "go":
            if entities["places"]:
                place = entities["places"][0]
                components.append(f"go to the {place}")
                actions_used.add("go")
        
        elif action == "play":
            if "friend" in text or "friends" in text:
                components.append("play with friends")
            elif entities["objects"]:
                obj = entities["objects"][0]
                components.append(f"play with the {obj}")
            else:
                components.append("play")
            actions_used.add("play")
        
        elif action == "eat":
            if entities["objects"]:
                obj = entities["objects"][0]
                components.append(f"eat {obj}")
            else:
                components.append("eat food")
            actions_used.add("eat")
        
        elif action == "drink":
            if entities["objects"]:
                obj = entities["objects"][0]
                components.append(f"drink {obj}")
            else:
                components.append("drink water")
            actions_used.add("drink")
        
        elif action == "sleep":
            components.append("sleep")
            actions_used.add("sleep")
        
        elif action == "read":
            if entities["objects"]:
                obj = entities["objects"][0]
                components.append(f"read a {obj}")
            else:
                components.append("read")
            actions_used.add("read")
        
        elif action == "watch":
            if entities["objects"]:
                obj = entities["objects"][0]
                components.append(f"watch {obj}")
            else:
                components.append("watch")
            actions_used.add("watch")
        
        elif action == "draw":
            if entities["objects"]:
                obj = entities["objects"][0]
                components.append(f"draw a {obj}")
            else:
                components.append("draw")
            actions_used.add("draw")
        
        elif action in ["want", "need", "like"]:
            # These are meta-actions, handle separately
            continue
    
    # Handle inferred intents
    if "implied_social_activity" in inferred_intents:
        place = entities["places"][0] if entities["places"] else "school"
        components.append(f"go to the {place} and play with friends")
    
    if "implied_rest" in inferred_intents:
        components.append("rest")
    
    if "implied_eat" in inferred_intents and "eat" not in actions_used:
        if entities["objects"]:
            obj = entities["objects"][0]
            components.append(f"eat {obj}")
        else:
            components.append("eat food")
    
    if "implied_drink" in inferred_intents and "drink" not in actions_used:
        if entities["objects"]:
            obj = entities["objects"][0]
            components.append(f"drink {obj}")
        else:
            components.append("drink water")
    
    # Handle emotions
    if entities["emotions"]:
        emotion = entities["emotions"][0]
        if not components:  # Only add emotion if no other components
            components.append(f"feel {emotion}")
    
    # Handle standalone objects (no actions detected)
    if not components and entities["objects"]:
        obj = entities["objects"][0]
        components.append(f"have {obj}")
    
    # Handle standalone places (no actions detected)
    if not components and entities["places"]:
        place = entities["places"][0]
        components.append(f"go to the {place}")
    
    return components


def combine_components_naturally(components):
    """
    Combine multiple components into a natural sentence.
    Handles 1, 2, or 3+ components gracefully.
    """
    if not components:
        return ""
    
    if len(components) == 1:
        return components[0]
    
    if len(components) == 2:
        return f"{components[0]} and {components[1]}"
    
    # 3 or more components
    return ", ".join(components[:-1]) + f" and {components[-1]}"


def apply_grammar_rules(sentence):
    """
    Apply comprehensive grammar rules to make sentence natural.
    """
    if not sentence:
        return sentence
    
    # Remove double spaces
    while "  " in sentence:
        sentence = sentence.replace("  ", " ")
    
    # Remove duplicate consecutive words
    words = sentence.split()
    deduped = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i-1]:
            deduped.append(word)
    sentence = " ".join(deduped)
    
    # Fix articles
    sentence = sentence.replace(" a apple", " an apple")
    sentence = sentence.replace(" a orange", " an orange")
    sentence = sentence.replace(" a egg", " an egg")
    
    # Fix "the the"
    sentence = sentence.replace("the the", "the")
    
    # Capitalize "I"
    sentence = sentence.replace(" i ", " I ")
    sentence = sentence.replace("i want", "I want")
    sentence = sentence.replace("i need", "I need")
    sentence = sentence.replace("i like", "I like")
    
    # Capitalize first letter
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    
    return sentence.strip()


def generate_sentence_variations(base_components, entities, text):
    """
    Generate 3 different natural variations of the sentence.
    Each variation uses different framing (want/like/need/feel).
    """
    if not base_components:
        return []
    
    combined = combine_components_naturally(base_components)
    
    variations = []
    
    # Variation 1: "I want to..."
    var1 = f"I want to {combined}"
    var1 = apply_grammar_rules(var1)
    variations.append(var1)
    
    # Variation 2: Different framing based on context
    if entities["emotions"]:
        emotion = entities["emotions"][0]
        var2 = f"I am {emotion} and want to {combined}"
    elif "go" in entities["actions"]:
        var2 = f"I would like to {combined}"
    else:
        var2 = f"I need to {combined}"
    var2 = apply_grammar_rules(var2)
    variations.append(var2)
    
    # Variation 3: Present continuous or different phrasing
    if "play" in combined:
        var3 = combined.replace("play", "playing")
        var3 = f"I feel like {var3}"
    elif "go" in combined:
        var3 = combined.replace("go to", "going to")
        var3 = f"I am thinking of {var3}"
    else:
        var3 = f"I wish to {combined}"
    var3 = apply_grammar_rules(var3)
    variations.append(var3)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)
    
    # Ensure exactly 3 variations
    if len(unique_variations) < 3:
        # Add fallback variations
        fallbacks = [
            f"I want to {combined}",
            f"Can I {combined}?",
            f"Please let me {combined}"
        ]
        for fb in fallbacks:
            fb = apply_grammar_rules(fb)
            if fb not in unique_variations:
                unique_variations.append(fb)
            if len(unique_variations) >= 3:
                break
    
    return unique_variations[:3]


def predict_with_model(sentence, max_len=15):
    """
    Use the trained Seq2Seq model to predict corrected sentence.
    Used as fallback when intent detection yields insufficient results.
    """
    tokens = numericalize(sentence)
    tokens = tokens[:max_len]
    tokens += [word2idx["<pad>"]] * (max_len - len(tokens))
    
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        hidden = model.encoder(src_tensor)
        input_token = torch.tensor([word2idx["<sos>"]]).to(device)
        result = []
        
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            top1 = output.argmax(1)
            
            if top1.item() == word2idx["<eos>"]:
                break
            
            word = idx2word.get(top1.item(), "")
            if word:
                result.append(word)
            
            input_token = top1
    
    return " ".join(result)


def generate_intelligent_suggestions(user_input):
    """
    Main function to generate 3 intelligent sentence suggestions.
    Uses multi-stage approach:
    1. Typo correction
    2. Entity detection
    3. Intent inference
    4. Component building
    5. Natural sentence generation
    6. Model fallback if needed
    """
    # Stage 1: Clean and correct typos
    cleaned = clean_input(user_input)
    corrected = apply_typo_corrections(cleaned)
    
    # Stage 2: Detect entities
    entities = detect_entities(corrected)
    
    # Stage 3: Infer implicit intents
    inferred = infer_implicit_intents(entities, corrected)
    
    # Stage 4: Build sentence components
    components = build_sentence_components(entities, corrected, inferred)
    
    # Stage 5: Generate variations
    if components:
        suggestions = generate_sentence_variations(components, entities, corrected)
    else:
        # Fallback to model if no clear intent detected
        model_output = predict_with_model(corrected)
        model_output = apply_grammar_rules(model_output)
        
        if model_output and len(model_output.split()) >= 3:
            suggestions = [
                model_output,
                "I need help",
                "Can you help me?"
            ]
        else:
            # Ultimate fallback
            suggestions = [
                "I want to say something",
                "I need help",
                "Can you help me?"
            ]
    
    # Ensure exactly 3 unique suggestions
    unique = []
    seen = set()
    for s in suggestions:
        if s not in seen:
            unique.append(s)
            seen.add(s)
    
    # Pad if needed
    while len(unique) < 3:
        fallbacks = [
            "I need assistance",
            "Please help me",
            "I want to tell you something"
        ]
        for fb in fallbacks:
            if fb not in seen:
                unique.append(fb)
                seen.add(fb)
                break
    
    return unique[:3]


# ===========================================================================
# EVALUATION AND METRICS
# ===========================================================================

def evaluate_accuracy_fast():
    """Calculate word-level accuracy on test dataset"""
    model.eval()
    total_score = 0
    total_words = 0
    
    with torch.no_grad():
        for inp, actual in test_data[:1000]:
            pred = predict_with_model(inp)
            
            pred_words = pred.split()
            actual_words = actual.split()
            
            for p, a in zip(pred_words, actual_words):
                if p == a:
                    total_score += 1
                total_words += 1
    
    return total_score / total_words if total_words > 0 else 0.0


def save_metrics(accuracy):
    """Save accuracy metrics to JSON file"""
    metrics = {
        "accuracy": accuracy,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples": min(1000, len(test_data))
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✓ Metrics saved to metrics.json")


def save_confusion_data():
    """Save confusion matrix data for analysis"""
    y_true = []
    y_pred = []
    
    for inp, actual in test_data[:1000]:
        pred = predict_with_model(inp)
        
        actual_words = actual.split()
        pred_words = pred.split()
        
        # Align predictions with actuals
        for i in range(max(len(actual_words), len(pred_words))):
            if i < len(actual_words):
                y_true.append(actual_words[i])
            else:
                y_true.append("<pad>")
            
            if i < len(pred_words):
                y_pred.append(pred_words[i])
            else:
                y_pred.append("<pad>")
    
    confusion_data = {
        "y_true": y_true,
        "y_pred": y_pred,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("confusion_data.json", "w") as f:
        json.dump(confusion_data, f, indent=4)
    
    print(f"✓ Confusion data saved to confusion_data.json")


def save_interaction(user_input, suggestions, selected=None):
    """Log user interactions for future analysis"""
    log_file = "interaction_log.json"
    
    new_entry = {
        "input": user_input,
        "suggestions": suggestions,
        "selected": selected,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Load existing log or create new
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        data = []
    
    data.append(new_entry)
    
    with open(log_file, "w") as f:
        json.dump(data, f, indent=4)


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("PACEWISE - ASD Communication Assistant")
    print("="*70)
    print("\nInitializing system...\n")
    
    # Calculate and save metrics
    print("Evaluating model accuracy...")
    accuracy = evaluate_accuracy_fast()
    print(f"✓ Word Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    save_metrics(accuracy)
    
    print("Generating confusion matrix data...")
    save_confusion_data()
    
    print("\n" + "="*70)
    print("System ready! Enter fragmented sentences to get suggestions.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*70 + "\n")
    
    # Main interaction loop
    interaction_count = 0
    
    while True:
        try:
            # Get user input
            print(f"\n[Session {interaction_count + 1}]")
            user_input = input("Enter fragmented sentence: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "stop", "bye"]:
                print("\n✓ Thank you for using PACEWISE. Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                print("⚠ Please enter some text.")
                continue
            
            # Generate suggestions
            print("\nProcessing...")
            suggestions = generate_intelligent_suggestions(user_input)
            
            # Display suggestions
            print("\n" + "-"*70)
            print("Suggestions:")
            print("-"*70)
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
            print("-"*70)
            
            # Get user choice
            choice_input = input("\nChoose best suggestion (1/2/3) or press Enter to skip: ").strip()
            
            selected = None
            if choice_input in ["1", "2", "3"]:
                idx = int(choice_input) - 1
                selected = suggestions[idx]
                print(f"✓ Selected: {selected}")
            else:
                print("⊘ No selection made")
            
            # Log interaction
            save_interaction(user_input, suggestions, selected)
            interaction_count += 1
            
        except KeyboardInterrupt:
            print("\n\n✓ Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            print("Please try again.")
    
    print(f"\nTotal interactions: {interaction_count}")
    print("All interactions logged to interaction_log.json")


if __name__ == "__main__":
    main()