# Multi-class tag list
TAGS = ["Billing", "Technical Support", "Account Recovery", "Cancellation", "General Inquiry"]

# 1. Sample Tickets
tickets = [
    "I was charged twice for my subscription this month, please refund.",
    "My screen goes completely black when I try to open the dashboard app.",
    "I forgot my password and my backup email is no longer active."
]

# 2. Zero-Shot Prompting Simulator
def zero_shot_classifier(ticket):
    ticket_lower = ticket.lower()
    scores = {tag: 0 for tag in TAGS}
    
    # Matching simple semantic keywords
    if "charge" in ticket_lower or "refund" in ticket_lower or "price" in ticket_lower:
        scores["Billing"] += 3
    if "black" in ticket_lower or "app" in ticket_lower or "bug" in ticket_lower or "error" in ticket_lower:
        scores["Technical Support"] += 3
    if "password" in ticket_lower or "email" in ticket_lower or "login" in ticket_lower:
        scores["Account Recovery"] += 3
        
    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [tag for tag, score in sorted_tags[:3]]

# 3. Few-Shot Prompting Simulator (Provides higher accuracy with examples)
FEW_SHOT_EXAMPLES = {
    "Technical Support": ["My app crashes", "The interface is lagging", "Cannot load dashboard"],
    "Billing": ["Need invoice", "Charged too much", "Payment failed"]
}

def few_shot_classifier(ticket):
    ticket_lower = ticket.lower()
    scores = {tag: 0 for tag in TAGS}
    
    # Primary check with few-shot knowledge mapping
    for tag, examples in FEW_SHOT_EXAMPLES.items():
        for example in examples:
            if example.lower() in ticket_lower or any(word in ticket_lower for word in example.lower().split()):
                scores[tag] += 2
                
    # Fallback to zero-shot
    fallback = zero_shot_classifier(ticket)
    for i, tag in enumerate(fallback):
        scores[tag] += (3 - i)
        
    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [tag for tag, score in sorted_tags[:3]]

# Run evaluations
print("--- Zero-Shot vs Few-Shot Ticket Tagging ---")
for t in tickets:
    print(f"\nTicket: '{t}'")
    print(f"Zero-Shot Tags: {zero_shot_classifier(t)}")
    print(f"Few-Shot Tags (Optimized): {few_shot_classifier(t)}")
