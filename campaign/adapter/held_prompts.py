"""Source texts of the HIDDEN held-out workload (prompts the dev set never uses; different shape: structured output, code, a 3K-token document, reasoning).
prepare.py freezes them (tokenised with the chat template) into <root>/private/held_prompts.json; nothing here is copied into an agent's tree.
MARKERS: distinctive strings of these texts; prepare.py adds them to the config's `forbidden` list so a diff that mentions them is reverted."""
MARKERS = ["ZQ-LEDGER-88", "ZQ-ROPE-14", "vx-4471", "Hollenbeck", "orders_2024q3_archive", "order_items_zq", "customers_zq", "Ana, Bo, Cy, Di", "hollenbeck.example"]

TEXTS = {
    "jsontool": (
        "You are an API orchestration assistant. Available functions:\n"
        "create_invoice(customer_id: str, line_items: list[{sku: str, qty: int, unit_price_cents: int}], currency: str, due_in_days: int, memo: str | null)\n"
        "send_email(to: str, subject: str, body: str)\n\n"
        "Customer vx-4471 (Hollenbeck Marine Supply) bought 12 units of SKU ZQ-LEDGER-88 at 1999 cents each and 3 units of SKU ZQ-ROPE-14 at 450 cents each, "
        "billed in CAD, due in 45 days, memo 'Q3 restock, net-45'. Then notify billing@hollenbeck.example that the invoice was created.\n"
        'Respond with only the two tool calls, each as a JSON object {"name": ..., "arguments": {...}}, one per line.'
    ),
    "sql": (
        "Schema (PostgreSQL): orders_2024q3_archive(order_id bigint, cust_id bigint, placed_at timestamptz, total_cents bigint, region_code text), "
        "order_items_zq(order_id bigint, sku text, qty int, discount_pct numeric), customers_zq(cust_id bigint, tier text, signup_date date).\n"
        "Write one query that returns, for each region_code, the top 3 customers by net revenue (item prices after discount_pct) over the 90 days ending at the latest placed_at, "
        "with their tier and rank, using CTEs and window functions. Then say which indexes you would add and why, in at most five lines."
    ),
    "reason": (
        "Five friends - Ana, Bo, Cy, Di and Eli - sit in a row of five seats numbered 1 to 5 from left to right. Clues: Ana sits somewhere to the left of Bo. "
        "Cy sits in seat 3. Di does not sit next to Cy. Eli sits next to Bo. Bo is not in seat 5. Who sits in each seat? "
        "Reason step by step, check every clue against your answer, then state the arrangement."
    ),
}
# "docqa": built by prepare.py = the first DOC_TOKENS tokens of the papers/*.md reading notes (not the topics/*.md used by the dev set's 8K prompt) + this question
DOC_TOKENS = 3000
DOC_QUESTION = "\n\nAccording to the notes above, name the three papers whose claims are most in tension with each other and explain the tension in two sentences each, quoting a sentence of the notes for each."
