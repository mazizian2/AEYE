from crewai import Task
import json
from agent.SHAgent import greeting_responder_agent, combined_need_agent, buy_create_json_agent, buy_responder_agent, \
    objection_handler_agent, context_switch_agent, register_order_json_agent, register_order_agent, \
    user_info_collector_agent, user_info_json_agent, unknown_agent, responder_question_suggest_agent, payment_agent
from general.tools import OUTPUT_HTML, extract_unique_values

with open("assets/json/internet_plans.json", "r", encoding="utf-8") as f:
    services_data = json.load(f)
typeService = extract_unique_values("assets/json/internet_plans.json", "نوع سرویس")

INTENTS = [
    "greeting",
    "express_need_buy",
    "extract_json_buy",
    "service_suggestion_buy",
    "answer_question_service_suggestion",
    "set_history_suggestion",
    "objection_buy",
    "register_order",
    "extract_user_info_json",
    "payment",
    "follow_up_buy",
    "support",
    "ask_problem",
    "set_problem",
    "get_info_account",
    "extract_json_account",
    "ask_witch_account",
    "extract_select_account",
    "unknown"
]


def context_switch_task(user_message: str, state: dict, last_ai_message: str, assistant_messages: list) -> Task:
    print("context_switch_task", state.get('intents'))
    history_suggestion = state.get("history_suggestion")
    previous_intents = state.get('intents', [])
    next_node = state.get('next_node', 'none')
    info = state.get('userInfo').get('info')
    accounts = state.get('userInfo').get('accounts')
    problems = state.get('userInfo').get('problems')
    selectedAccount = state.get('userInfo').get('selectAccount')
    return Task(
        description=(
            "You must decide which path the new user message belongs to based on the list of previous intents and the current message.\n"
            f"List of possible paths: {', '.join(INTENTS)}\n\n"
            f"Previous intents: {previous_intents}\n"
            f"Current path (next_node): {next_node}\n"
            f"User info (if available): {info}\n"
            f"Last AI message (if available): {last_ai_message if last_ai_message else 'none'}\n"
            f"Suggested services (if available): {history_suggestion}\n"
            f"New user message: {user_message}\n\n"
            f"User required values: {state.get('userInfo', {}).get('needs', {})}\n\n"
            f"User accounts (if available): {accounts}\n"
            f"User account selected (if available): {selectedAccount}\n"
            f"User problems (if available): {problems}\n"

            "Rules of detection:\n"
            "- If services were previously suggested (history_suggestion exists) and the new message asks about those services, output must be 'answer_question_service_suggestion'.\n"
            "- If the message continues the previous intent or includes phrases like 'explain more', 'I didn't understand', output should be the same previous intent.\n"
            "- If the user greets, output should be 'greeting'.\n"
            "- If the user asks about the brand or responsibilities of the assistant, output should be 'greeting'.\n"
            "- If the message includes both greeting and a service purchase need, priority goes to service purchase.\n"
            "- If the new message includes more details about the purchase need (like specifying LTE, ADSL, etc.), **or includes a selector word like 'the last one', 'the first one', 'آخری', 'اولی', 'همون' referring to a service type**, even if one word, and the last AI message was about service types, **and history_suggestion is empty**, output must be 'extract_json_buy'."
            # "- If the user message only contains the name of a service type (like LTE, ADSL, VDSL, TD-LTE, Fiber), even a single word, the output must be 'extract_json_buy'.\n"
            # "- If the user expresses a need to buy a service but doesn’t specify the type, output must be 'express_need_buy'.\n"
            "- If the user expresses a need to buy a service but doesn’t specify the type, OR if they only ask about the available service types (e.g., 'What services do you have?' or 'Do you offer ADSL service?'), output must be 'express_need_buy'."
            # "- If the user expresses a need to buy a service and gives details like type, speed, budget, or features, output must be 'extract_json_buy'.\n"
            "- If the user expresses a need to buy a service and provides details like type, speed, budget, or features, OR implicitly signals readiness to hear options based on their specific needs, output must be 'extract_json_buy'."
            "- If the user asks for service suggestions and is ready to hear options, output must be 'extract_json_buy'.\n"
            "- If the user objects or expresses concern about price or quality of a service, output must be 'extract_json_buy'.\n"
            "- **Only if** Suggested services (history_suggestion) is not empty and user confirms or selects one of the suggested services..."
            "- If the user shares personal info like name or phone number, output must be 'extract_user_info_json'.\n"
            "-If none of the values in (info) were empty, and orders also had at least one item, and the status of that item had the value 'payment successful', and the previous node was 'payment', the output must be follow_up_buy."
            "- If the topic of the last AI message is about obtaining account information (such as username, mobile number, national ID, or last name) AND the user's message contains at least one piece of account information (even a last name like 'دهدار'), output must be 'extract_json_account'. This rule has priority over others.\n"
            "- If the meaning of the user's message is support :\n"
            "    - If the user explicitly states a problem (like مشکل، خرابی، قطعی, وصل نمی‌شود), output must be 'set_problem'.\n"
            "    - Otherwise , output must be 'support'.\n"
            "- For messages about service types:\n"
            "  - If the last AI message contained service suggestions based on user need, output must be 'extract_json_buy'.\n"
            "  - Otherwise, output must be 'express_need_buy'.\n"
            "- If none of the rules apply, output 'unknown'.\n\n"
            "- اگر مفهوم پیام کاربر مربوط به مسائل پشتیبانی ( نیاز به پشتیبانی، خرابی سرویس، قطعی و...) باشد خروجی باید support باشد\n"
            "⚠️ Important Notes:\n"
            "- Detection must consider conversation context.\n"
            "- Continuity of topic matters in choosing the intent.\n"
            "- If the user message is short or a single word, use the last AI message to decide next node.\n"
            "- Output must be only one node name without any extra text and without quotes."
        ),

        agent=context_switch_agent,
        expected_output="One of the node names from the list"
    )


def create_greeting_response_task(user_message: str, ai_message: None, user_info=None):
    print("last_ai_message is>>>", ai_message)
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User info (if available): {user_info.get('info')}\n"
            f"Previous AI messages (if available): {ai_message}\n\n"
            "Generate a warm and engaging response in Persian (Farsi) based on the following rules:\n"
            "1. The response must continue the conversation naturally and must not repeat previous AI messages.\n"
            "2. Introduce the brand 'شبکیه' and its core purpose (internet service purchase) only in early conversations "
            "and only if it has not already been mentioned in previous AI messages.\n"
            "3. Avoid repeating greetings like 'سلام' if already used in earlier AI responses. Respond appropriately based on context.\n"
            "4. If the user's name is available, use it naturally for personalization.\n"
            "5. Maintain a friendly, trustworthy, and professional tone.\n"
            "6. Limited and professional emoji use is allowed and encouraged for a friendly tone.\n"
            "7. Close the message by expressing readiness to assist the user.\n\n"
            "Output Requirements:\n"
            "- The response must be returned as **clean raw HTML**.\n"
            "- Avoid repetitive content and robotic phrases.\n"
            "- Keep the message concise but meaningful.\n\n"
            "The answers must be in Persian, without exception."
            f"{OUTPUT_HTML}"

        ),

        agent=greeting_responder_agent,
        expected_output="Raw HTML text"
    )


def create_express_need_buy_response_task(user_message, state=dict):
    print("typeService is>>>", typeService)
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User info: {state.get('userInfo', {})}\n\n"
            f"User required values: {state.get('userInfo', {}).get('needs', {})}\n\n"
            f"Available service types: {typeService}\n\n"
            "If the user has not yet specified the service type they want, guide them "
            "gently to clarify it. The goal of this step is to ensure the user explicitly "
            "**chooses a service type**, as it is required to continue.\n\n"
            "Response Rules:\n"
            "- If the user asks about available services, list all services from "
            "`typeService` and provide a short, simple explanation for each one, then ask: "
            "‘Which one would you like to purchase?’\n"
            "- If `state['userInfo']['needs']['type']` is empty, encourage the user to specify "
            "their desired service type. Provide helpful examples from the available list "
            f"({', '.join(typeService)}), and if needed, ask what purpose they need it for "
            "— for example streaming, gaming, remote work, etc.\n"
            "- After identifying the service type, motivate the user to share important "
            f"specifications or preferences based on available options in {services_data}. "
            "If they don’t provide details, that’s fine — continue smoothly.\n"
            "- If `state['userInfo']['needs']['type']` is already filled, **do not ask about "
            "service type again**. Only focus on gathering service features and requirements.\n"
            "- Keep messages short, user-friendly, and helpful.\n"
            "- Avoid greetings or small talk if the user didn’t start with one. Do NOT use: "
            "greetings, introductions, jokes, emojis, or generic phrases like ‘How can I "
            "help you today?’\n\n"
            "The answers must be in Persian, without exception."
            f"{OUTPUT_HTML}"
        ),

        agent=buy_create_json_agent,
        expected_output="Raw HTML text"
    )


def create_json_need_buy_response_task(user_message, last_ai_message, state=None):
    assistant_messages = [m["content"] for m in state.get("messages", []) if m.get("role") == "assistant"]
    history_suggestion = state.get("history_suggestion")
    previous_json = state.get('userInfo', {}).get('needs', {})
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"Message history: {assistant_messages}\n\n"
            f"Last AI message (if any): {last_ai_message if last_ai_message else 'None'}\n"
            f"Suggested services (if any): {history_suggestion}\n\n"
            f"User information: {state.get('userInfo', {})}\n\n"
            f"User needs: {state.get('userInfo', {}).get('needs', {})}\n\n"
            f"Previous selections: {previous_json}\n"

            f"List of service types: {typeService}\n"
            f"JSON with service features: {services_data}\n\n"
            "Instructions:\n"
            "1. Precisely analyze and interpret the user message.\n"
            "2. Extract user needs into two parts:\n"
            "   a) Service type ('type') – must be selected only from typeService.\n"
            "   b) Feature details ('details') – extract based on keys and values in services_data.\n"
            "      - **New:** Include any feature, condition, explanation, or limitation mentioned by the user.\n"
            "- Never add extra items beyond what the user requested.\n"
            "- Do not hallucinate or guess user intent.\n"
            "- Merge new values with previous selections from 'previous_json', avoiding duplicates.\n"
            "- If the user explicitly rejects or limits any previous selection, remove only that specific item.\n"
            "- If the user provides no features or details, leave 'details' as an empty list.\n"
            # "- If the user provides no service type, include all items from typeService in 'type'.\n\n"
            " If the user provides no service type **and does not use a sequential reference (per rule 5)**, include all items from typeService in 'type'.\n\n"
            "3. If the message includes general terms matching multiple typeService items (e.g., 'LTE' or 'server'), include all related items in the output.\n"
            "4. If the user does not specify a service but mentions its use case, choose the best matching service features from services_data for 'details'. If no service type is mentioned, include all typeService in 'type'; otherwise, only the mentioned service.\n"
            "5. Selection guidelines (High Priority):\n"
            "- Identify the intended service from `last_ai_message` or `history_suggestion`.\n"
            "- Direct selection: If the user mentions the exact service name or unique details, select that service.\n"
            "- **Sequential references (Very Important):** If the user uses a sequential reference (e.g., 'اولی' (first), 'دومی' (second), 'آخری' (last), 'یکی مونده به آخری' (second to last), ...):\n"
            "    **a.** **Parse the list of offered services** from the `last_ai_message` (e.g., from `<ul><li>` tags or other list formats).\n"
            "    **b.** Calculate the index based on the formula:\n"
            "        * Positive order: 'اولی' (1), 'دومی' (2), ... → `index = n-1`\n"
            "        * Negative order: 'آخری' (1st from end), 'دومی از آخر' (2nd from end), ... → `index = -n`\n"
            "    **c.** **Extract the service name** from the parsed list using this index.\n"
            "    **d.** Set the 'type' list to contain **only** this extracted service name. (Example: If 'آخری' (last) refers to 'Adsl', the output must be `{'type': ['Adsl']}`).\n"
            "    **e.** This rule **overrides** the default behavior in rule 2. Do not add all services if a sequential reference is successfully resolved.\n\n"
            # "5.Selection guidelines:\n"
            # "- Identify the intended service from last_ai_message.\n"
            # "- Direct selection: If the user mentions the exact service name or unique details, select that service.\n"
            # "- Sequential references:\n"
            # "    * Positive order: «اولی», «دومی», «سومی», ... → index = n-1\n"
            # "    * Negative order: «آخری», «یکی مونده به آخری», «دومی از آخر», ... → index = -n\n"
            # "    * General formula: index = number-1 if positive, index = -number_from_end if negative\n"
            "6. Special rule for handling objections or adding constraints to previous user needs:\n"
            "If services were previously suggested and history_suggestion is not empty, take the last JSON from history_suggestion. If the user objects to or adds a constraint on a feature (e.g., price, speed, volume, ping):\n"
            "- Identify the feature in question.\n"
            "- If the feature is numeric (price, volume, speed, ping):\n"
            "  - Extract all values for this feature from previous suggestions.\n"
            "  - Always use the **highest value** among them.\n"
            "  - If a reduction is requested (e.g., 'cheaper', 'less'):\n"
            "      → Example: 'price below <highest suggested price>'\n"
            "  - If an increase is requested (e.g., 'faster', 'more'):\n"
            "      → Example: 'volume above <highest value>'\n"
            "  - Important: always use the largest available value, never the first or lowest.\n"
            "- If the feature is non-numeric (e.g., location, support type, IP type):\n"
            "  - Include it exactly as descriptive text:\n"
            "      'Location: Germany'\n"
            "      '24/7 Support'\n"
            "      'Static IP'\n\n"
            "- Always preserve previous 'details' and append new features unless explicitly negated.\n"
            "Very important: Only check the feature that was objected to or constrained, not other features.\n\n"
            "7. Avoid repetition and maintain logical order.\n"
            "8. Output must be only two valid Python lists, without extra text or explanation.\n"
            "9. Only extract service titles for 'type' and standard expressions for 'details'.\n\n"
            "10. Special rules for gaming or new needs:\n"
            "- If the user explicitly mentions a new need (e.g., 'I need to play games'):\n"
            "  → Add all features suitable for that use case to 'details'.\n"
            "- If the user only asks for clarification, follow-up, or continuation (e.g., 'What else do you have?', 'Just these?'):\n"
            "  → Do not clear or modify 'details'; keep previous selections.\n"
            "- Always merge new features with previous ones, avoiding duplicates.\n"
            "- Only apply changes or constraints if the user explicitly requests it; otherwise preserve all previous 'details'.\n"

            "Example output:\n"
            "{\n"
            "  'type': ['lte-MCI', 'lte-Irancell'],\n"
            "  'details': ['maximum high speed', 'includes modem', 'long duration']\n"
            "}\n"
            "⚠️ Output must be only the JSON lists without any extra text."
        ),
        agent=combined_need_agent,
        expected_output="Output is a JSON with keys 'type' and 'details', each containing a list of valid values"
    )


def create_service_suggestion_response_task(user_message, user_info=None, user_needs=None):
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User information: {user_info}\n"
            f"User needs: {user_needs}\n"
            f"Available services: {services_data}\n"
            "You should suggest all services that are closest and most similar to the user's needs.\n"
            "Only propose options that are relevant from the available services.\n"
            "Response guidelines:\n"
            "1. The first sentence of your answer should start with a sentence like:\n"
            "بر اساس نیازهایی که ذکر کرده‌اید، ..."
            "2. Only suggest relevant services from the available ones.\n"
            "3. Explain why each service is suitable (logical reasoning + real value).\n"
            "4.If 2 or more services match the user's needs, list every single suitable service. "
            "For each service, explain why it is suitable and highlight the differences between them clearly."
            "Do not skip any service that is relevant.\n"
            # "4. If multiple options are suitable, introduce all and explain their differences.\n"
            "5. The response must be clear and easy to understand.\n"
            "6. Never provide incorrect or imaginary information.\n"
            "7. Ensure suggested services are based on the user's needs and available in services_data.\n"
            "8. If no service matches the user's needs:\n"
            "- Politely inform the user that no exact match exists and apologize.\n"
            "- If there is a similar type of service, suggest it; otherwise, randomly pick two from services_data.\n"
            "- Explain why these can be temporary or alternative options for the user.\n"
            "- Include sentences like:\n"
            "  'Which services would you like me to activate?'\n"
            "  'I can also find a better suggestion if you specify the exact features you want.'\n"
            "⚠️ If the user has not greeted, do not include greetings, small talk, emojis, or general phrases like 'We are at your service.' Respond only if the user greets.\n"
            "The answers must be in Persian, without exception."
            f"{OUTPUT_HTML}"
        ),

        agent=buy_responder_agent,
        expected_output="Raw HTML text"
    )


def create_service_suggestion_json_task(ai_message: str):
    return Task(
        description=(
            f"Message containing suggested services: {ai_message}\n"
            f"Available service types: {typeService}\n"
            f"Service database: {services_data}\n\n"
            "Selection instructions:\n"
            "1. Consider the message containing suggested services (ai_message).\n"
            "2. Extract all services mentioned along with all their attributes.\n"
            "3. Use only data from services_data. Do not guess or fabricate any values.\n\n"
            "Output requirements:\n"
            "- Return a list containing only the services suggested in ai_message.\n"
            "- The output must strictly follow the JSON schema from services_data.\n"
            "- Do not include any additional text or commentary.\n\n"
            "Correct output example:\n"
            "[\n"
            "  {\n"
            '    \"نوع سرویس\": \"FD-LTE همراه اول\",\n'
            '    \"نام سرویس\": \"جشنواره پاییزه 100 روزه +100 گیگ+ 144 گیگ شبانه\",\n'
            '    \"حداقل سرعت\": 1,\n'
            '    \"حداکثر سرعت دانلود\": 50,\n'
            '    \"حداکثر سرعت آپلود\": 5,\n'
            '    \"سطح پوشش دهی\": \"کشوری\",\n'
            '    \"مودم\": \"ندارد\",\n'
            '    \"شبانه\": 144,\n'
            '    \"ترافیک\": 100,\n'
            '    \"زمان\": 100,\n'
            '    \"IP\": \"ندارد\",\n'
            '    \"قیمت\": 3355000\n'
            "  }\n"
            "]\n\n"
            "⚠️ Only return the list with JSON data. No extra text is allowed."
        ),
        agent=register_order_json_agent,
        expected_output="A valid Python list of JSON objects representing the suggested services."
    )


def create_answer_service_suggestion_response_task(user_message, state=None):
    user_info = state['userInfo'] if 'userInfo' in state else None
    user_needs = state['userInfo']['needs']
    assistant_messages = [m["content"] for m in state.get("messages", []) if m.get("role") == "assistant"]
    history = state["history_suggestion"]
    history_suggestion = history[-1]
    print('history_suggestion isssssssssssssssssssssssssssssss>>>', history_suggestion)
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User info: {user_info}\n"
            f"User needs: {user_needs}\n"
            f"Assistant messages: {assistant_messages}\n"
            f"Previously suggested services: {history_suggestion}\n"
            f"Service features JSON: {services_data}\n\n"
            "Instructions for the response:\n"
            "1. Understand the user's question.\n"
            "2. Answer ONLY based on previously suggested services (history_suggestion).\n"
            "   - Do NOT suggest any new services.\n"
            "   - Do NOT provide general advice unrelated to history_suggestion.\n"
            "   - Provide concise, clear, and factual information relevant to the question.\n"
            "Selection guidelines:\n"
            "3 Identify the intended service from history_suggestion.\n"
            "- Direct selection: If the user mentions the exact service name or unique details, select that service.\n"
            "- Sequential references:\n"
            "    * Positive order: «اولی», «دومی», «سومی», ... → index = n-1\n"
            "    * Negative order: «آخری», «یکی مونده به آخری», «دومی از آخر», ... → index = -n\n"
            "    * General formula: index = number-1 if positive, index = -number_from_end if negative\n"
            "4. If the question cannot be answered from history_suggestion, politely indicate that.\n"
            "5. Tone: instructive, polite, clear. Avoid repetition, filler, greetings, jokes\n\n"
            "The answers must be in Persian, without exception."
            f"{OUTPUT_HTML}"
        ),
        agent=responder_question_suggest_agent,
        expected_output="Raw HTML text"
    )


def register_order_json_task(user_message, state=None):
    assistant_messages = [m["content"] for m in state.get("messages", []) if m.get("role") == "assistant"]
    history = state.get("history_suggestion")
    orders = state.get("userInfo").get("orders")
    history_suggestion = history[-1]
    print('last of history_suggestion is>>>>', history_suggestion)
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"Message history: {assistant_messages}\n\n"
            f"Suggested services list: {history_suggestion}\n\n"
            f"User information: {state.get('userInfo', {})}\n"
            f"User needs: {state.get('userInfo', {}).get('needs', {})}\n\n"
            f"Service types list: {typeService}\n"
            f"Services database (services_data): {services_data}\n\n"
            "Goal:\n"
            "At this step, the user intends to select one of the suggested services. "
            "Your task is to extract the service selected by the user from the latest suggested services.\n\n"
            "Guidelines and selection logic:\n"
            # "- First, find the last AI message from assistant_messages that contained the service suggestion. "
            "Find it from the suggested services list history_suggestion.\n"
            "- Then check which service the user intends to choose in the current message.\n"
            "- If the message is a direct selection of a service (like the exact service name or part of its details), select that service.\n"
            "Selection guidelines:\n"
            "- Identify the intended service from history_suggestion.\n"
            "- Direct selection: If the user mentions the exact service name or unique details, select that service.\n"
            "- Sequential references:\n"
            "    * Positive order: «اولی», «دومی», «سومی», ... → index = n-1\n"
            "    * Negative order: «آخری», «یکی مونده به آخری», «دومی از آخر», ... → index = -n\n"
            "    * General formula: index = number-1 if positive, index = -number_from_end if negative\n"
            "- Relative references: «همونی که قبلاً گفتی», «اون ارزونه», «بهتره» → match based on history_suggestion and userInfo. "
            "If unclear, return an empty list.\n"
            "- Single suggestion: If only one service was suggested, select it automatically.\n"
            "- Never add duplicate data.\n"
            "Final output:\n"
            "You must return only a list containing **exactly one service** selected by the user. "
            "- If the user mentions more than one service, select **only the first one** based on order in history_suggestion."
            "- Service information must be exactly according to the user’s choice and extracted from services_data. No guessing or fabricated values are allowed."
            "- If the user’s choice is unclear, return an empty list."
            "Correct output example:\n"
            "[\n"
            "  {\n"
            '    \"نوع سرویس\": \"FD-LTE همراه اول\",\n'
            '    \"نام سرویس\": \"جشنواره پاییزه 100 روزه +100 گیگ+ 144 گیگ شبانه\",\n'
            '    \"حداقل سرعت\": 1,\n'
            '    \"حداکثر سرعت دانلود\": 50,\n'
            '    \"حداکثر سرعت آپلود\": 5,\n'
            '    \"سطح پوشش دهی\": \"کشوری\",\n'
            '    \"مودم\": \"ندارد\",\n'
            '    \"شبانه\": 144,\n'
            '    \"ترافیک\": 100,\n'
            '    \"زمان\": 100,\n'
            '    \"IP\": \"ندارد\",\n'
            '    \"قیمت\": 3355000\n'
            "  }\n"
            "]\n\n"
            "⚠️ The output must be only the list with JSON data without any extra text."
        ),
        agent=register_order_json_agent,
        expected_output="List of available services"
    )


def create_objection_buy_response_task(user_message, user_info=None):
    return Task(
        description=(
            f"پیام کاربر: {user_message}\n"
            f"اطلاعات کاربر: {user_info}\n"
            "راهنمای پاسخ‌دهی به اعتراض:\n"
            "1. ابتدا اعتراض یا نگرانی مشتری را بازگو کن تا مطمئن شود شنیده شده.\n"
            "2. سپس با دلیل منطقی، توضیح فنی یا مثال عملی نگرانی او را رفع کن.\n"
            "3. اگر نیاز شد، گزینه‌های جایگزین یا پیشنهادهای اضافی ارائه بده.\n"
            "4. هرگز مشتری را مجبور به خرید نکن یا فشار وارد نکن.\n"
            "5. لحن دوستانه، صمیمی و حرفه‌ای باشد.\n"
            "6. در پایان پیام، مکالمه را باز نگه دار و یک سوال ادامه‌دهنده بپرس "
            "(مثلاً «دوست دارید بیشتر درباره گزینه X بدانید؟»)\n"
            "⚠️این موارد ممنوع است: سلام کردن، احوال‌پرسی، معرفی، شوخی.\n"

            f"{OUTPUT_HTML}"
        ),
        agent=objection_handler_agent,
        expected_output="متن HTML خام"
    )


def register_order_response_task(user_message, state=None):
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User info: {state.get('userInfo')}\n"
            f"User name: {state.get('userInfo').get('info').get('name')}\n"
            f"Purchased services: {state.get('userInfo', {}).get('orders', {})}\n"
            "Response guidelines:\n"
            "1. Start with a warm and respectful message addressing the user by name, and thank them for their purchase.\n"
            "2. Then provide a concise and clear summary of the purchased services.\n"
            "3. Close the conversation with a friendly and pleasant sentence.\n"
            "4. The entire response should be concise, professional, and friendly.\n\n"
            "Example response: 'Thank you for your purchase! You have ordered: [cart summary]. "
            "We appreciate your trust and look forward to seeing you again!'\n"
            "⚠️Prohibited: greetings, small talk, self-introduction, jokes.\n"
            "The answers must be in Persian, without exception."
            f"{OUTPUT_HTML}"

        ),

        agent=register_order_agent,
        expected_output="Raw HTML text"
    )


def user_info_collector_response_task(user_message, state=None):
    name = state.get('userInfo').get("info").get('name')
    phone = state.get('userInfo').get("info").get('phone')
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User name: {name}\n"
            f"User phone number: {phone}\n"
            "Inform the user that to continue, some information needs to be completed.\n"
            # "Respectfully assure the user that their information will remain secure.\n"
            "Response rules:\n"
            "1. Respond politely and in a friendly tone, keeping the conversation going.\n"
            "2. Ask the appropriate question based on the stored information:\n"
            "   - If both name and phone are empty: ask a question including both.\n"
            "     Example: 'Please provide your full name and phone number.'\n"
            "   - If only name is empty: ask only for the full name.\n"
            "   - If only phone is empty: ask only for the phone number.\n"
            "3. Phone number must be valid (an 11-digit Iranian number starting with 09).\n"
            "   If the number is invalid, politely request a correct number.\n"
            "4. Avoid asking additional or irrelevant questions.\n"
            "5. Tone: polite, friendly, and clear.\n"
            "⚠️ Forbidden: greetings, small talk, self-introduction, jokes, emojis, or generic sentences like 'we are at your service'.\n"
            "The answers must be in Persian, without exception."
            f"{OUTPUT_HTML}"
        ),
        agent=user_info_collector_agent,
        expected_output="Raw HTML text"
    )


def user_info_json_task(user_message, state=None):
    info = state["userInfo"]['info']
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User information: {info}\n"
            "Carefully analyze the user message and extract the user's full name and valid phone number.\n\n"
            "Guidelines:\n"
            "- Extract any information containing the user's full name from the message and place it in the JSON.\n"
            # "- If the user's phone number exists in the message and is valid (an 11-digit Iranian number starting with 09), add it to the JSON.\n"
            "- Only extract valid Iranian phone numbers that:\n"
            "  1. Start with '09'\n"
            "  2. Have exactly 11 digits\n"
            " - Ignore any numbers that fall outside these rules\n"
            # "Important note: if info contains non-empty values for name or phone, consider the following rules:\n"
            "Important:\n"
            "- If info contains a non-empty value for name or phone and no new value is found in the user message, keep the info value in the output.\n"
            "- If a new value for name or phone is found in the user message, replace the previous value with the new one.\n"
            "- If neither the message nor info contains a valid value, leave the field empty ('').\n\n"
            "Example output:\n"
            "{'name':'علی مردادی','phone':'09134698765'}\n"
            "⚠️ The output must contain only JSON data without any extra text."
        ),
        agent=user_info_json_agent,
        expected_output="JSON with keys name and phone"
    )


def payment_task(user_message, state=None):
    order = state["userInfo"]['orders'][-1]
    info = state["userInfo"]['info']
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User order: {order}\n"
            f"User info: {info}\n\n"
            "Response Guidelines:\n"
            "1. Start with a warm, professional, and respectful greeting that addresses the user by name, "
            "and thank them for their order.\n"
            "2. Provide a short summary of the services or products in the user's cart.\n"
            "3. Clearly state that to finalize the order, they need to proceed with the payment using "
            "the link below:\n"
            '<a href="http://127.0.0.1:8000/pay" target="_blank">Click here to complete your payment</a>\n'
            "4. Show the total price including a 10% tax (add 10% to the original order price and display it clearly).\n"
            "5. Encourage the user to complete the payment to activate and finalize their order.\n"
            "6. The output should be raw HTML text only—no markdown, no fake data, and no fake links. "
            "Use the provided payment link exactly as given.\n\n"
            "Tone Requirements:\n"
            "- Friendly and professional\n"
            "- Clear and encouraging\n"
            "- Trustworthy and helpful\n"
            "The answers must be in Persian, without exception."
            "⚠️Prohibited: greetings, small talk, self-introduction, jokes.\n"

        ),
        agent=payment_agent,
        expected_output="Raw HTML text"
    )


def unknown_response_task(user_message):
    return Task(
        description=(
            f"User message: {user_message}\n"
            "1. Review the message received from the user."
            "   a. Politely and professionally inform the user that you did not fully understand their message."
            "   b. Ask the user to explain their goal or need more precisely so you can guide them better."
            "2. Always maintain a polite, clear, and friendly tone in your response."
            "⚠️ These are prohibited: greetings, small talk, introductions, jokes, emojis, or general phrases like 'we are at your service'.\n"
            "The answers must be in Persian, without exception."
            f"{OUTPUT_HTML}"
        ),

        agent=unknown_agent,
        expected_output="Raw HTML text"

    )
