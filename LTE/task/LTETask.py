from crewai import Task
import json
from LTE.agent.LTEAgent import ask_problem_agent,set_problem_agent, get_info_account_user_agent, extract_info_accounts_agent, \
    ask_witch_account_agent, extract_select_account_agent
from general.tools import OUTPUT_HTML
from datetime import datetime

INTENTS = [
    "support",
    "buy_service",
    "unknown",
    "greeting",
]
SUBINTENTS = [
    "help_internet",
    "detect_user_info",
    "request_collect_user_info",
    "collect_user_info",
    "ask_which_account",
    "detect_select_account",
    "check_modem",
    "turn_on_modem",
    "setting_modem",
    "edit_setting_modem",
    "return_operator",
]

def create_ask_problem_task(user_info):
    return Task(
        description=(
            f"User info {user_info}"
            "Ask the user, in a friendly and informal tone, to describe their internet problem.\n"
            "If the user's name is available, use it naturally for personalization.\n"
            "### Response Style Rules:\n"
            "- Use casual, friendly language.\n"
            "- Use emojis or stickers where appropriate.\n"
            "- If the user greeted you, greet them back politely.\n"
            "- If the user did NOT greet, do NOT greet. Go straight to asking about the problem.\n"
            "- Keep the message short and natural.\n"
            "- Always respond in the SAME language as the user.\n"
            "- Focus ONLY on internet and Shabakiye-related issues.\n\n"
             "⚠️ Prohibited: Greeting, small talk, or self-introduction.\n"
            f"{OUTPUT_HTML}"

        ),
        agent=ask_problem_agent,
        expected_output="Raw HTML text"
    )

def create_problem_list_task(user_message):
    return Task(
        description=(
            f"User message: {user_message}\n"
            "### Response Instructions:\n"
            "- Output ONLY a clean bullet-point list using hyphens (,).\n"
            "- Each bullet must be concise, clear, and describe a single problem.\n"
            "- Do NOT provide solutions, explanations, or extra text.\n"
            "- ALWAYS respond in the exact same language as the user.\n"
            "- If the message is unrelated to internet or 'Shabkiyeh' services, return an empty list.\n\n"
            "### Strict Output Format:\n"
            "[\n"
            "  'problem1',\n"
            "  'problem2',\n"
            "  ...\n"
            "]\n\n"
            "### Examples:\n"
            "User: My internet speed is very slow and sometimes disconnects.\n"
            "Assistant Response:\n"
            "[\n"
            "  'Slow internet speed',\n"
            "  'Intermittent disconnection'\n"
            "]\n\n"
            "User: سرعت اینترنت من خیلی کند است و گاهی قطع می‌شود.\n"
            "Assistant Response:\n"
            "[\n"
            "  'سرعت اینترنت پایین',\n"
            "  'قطع و وصل شدن گاه به گاه'\n"
            "]\n\n"
            "User: I want to know about cable TV packages.\n"
            "Assistant Response:\n"
            "[]"
        ),
        agent=set_problem_agent,
        expected_output="A list of string problems"
    )


def get_account_user_task(state):
    user_message = state["input"]
    info_accounts = state["userInfo"]["accountInfo"]
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User account info: {info_accounts}\n"
            "### Instructions for LLM:\n"
            "1. If all fields are empty: list username, mobile, last name, national ID with format rules "
            "and ask the user to provide at least one.\n"
            "2. If some fields are filled: no need to ask again.\n"
            "3. If any field is invalid: identify the field, show correct format, request correct data.\n"
            "4. Always respond in the user's language (Persian or English).\n"
            "5. Tone: polite, friendly, and instructive.\n"
            "⚠️ Forbidden: greetings, chit-chat, self-introduction, jokes.\n"
            f"{OUTPUT_HTML}"
            # f"User message: {user_message}\n"
            # f"User info account: {info_accounts}\n"
            # "پرسش از کاربر برای ارسال اطلاعات حساب کاربری با توجه به  info account"
            # "### Response Rules:\n"
            # "اگر تمامی مقادیر موجود در info_accounts خالی باشد، "
            # "باید موارد زیر را برای کاربر عنوان کنید:\n"
            # " نام کاربری، قانون :  شروع با 989559 و 12 رقم\n "
            # " شماره موبایل ، قانون :  شروع با 09 و 11 رقم \n"
            # " نام خانوادگی \n"
            # " کدملی، قانون :  کد ده رقمی \n"
            # "و از او بخواهید حداقل یکی از اطلاعاتش را ارسال کند."
            # "اگر مقادیر موجود در info_accounts خالی نباشد:\n"
            # "لازم نیست سوالی پرسیده شود \n"
            # "اما اگر داده نادرستی در هرکدام از فیلد های کدملی (melicode) ، شماره موبایل (mobile) ، نام کاربری(username) ، نام خانوادگی، با توجه به قوانین شان وجود داشت:\n"
            # "باید اعلام کنید که داده اشتباه در کدام فیلد هست و روش صحبحش رو اعلام کنید و بخواهید داده درست ارسال کند.\n"
            # "پیام باید حتما با توجه به پیام کاربر فارسی یا انگلیسی باشد\n"
            # "لحن مودبانه صمیمانه و راهنماکننده باشد.\n"
            # "⚠️این موارد ممنوع است: سلام کردن، احوال‌پرسی، معرفی، شوخی.\n"
            # f"{OUTPUT_HTML}"

        ),
        agent=get_info_account_user_agent,
        expected_output="Raw HTML text"
    )


def create_json_account_task(state):
    user_message = state["input"]
    info_accounts = state["userInfo"]["accountInfo"]
    return Task(
        description=(
            f"User message: {user_message}\n"
            "Analyze the user message and extract a valid JSON according to the following rules:\n"
            "- If the message contains a 12-digit code starting with 989559, treat it as 'username'.\n"
            "- If the message contains a 10-digit code, treat it as 'melicode'.\n"
            "- If the message contains an 11-digit code starting with 09, treat it as 'mobile'.\n"
            "- If the message contains the user's last name, treat it as 'lastname'.\n\n"
            "Example 1:\n"
            "User message: مرادی\n"
            "Output:\n"
            "{'lastname': 'مرادی', 'username': '', 'mobile': '', 'melicode': ''}\n\n"
            "Example 2:\n"
            "User message: 989559143561\n"
            "Output:\n"
            "{'username': '989559143561', 'lastname': '', 'mobile': '', 'melicode': ''}\n\n"
            "Note: If the message contains multiple pieces of information matching these rules, extract all of them and include them in the JSON.\n"
            "⚠️ The output must contain only JSON data without any extra text."
        ),
        agent=extract_info_accounts_agent,
        expected_output="JSON with keys melicode and mobile and username and lastname"
    )


def ask_witch_account_task(state):
    user_message = state["input"]
    accounts = state["userInfo"]["accounts"]
    selected_account = state["userInfo"]["selectAccount"]
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User accounts: {accounts}\n"
            f"Selected account: {selected_account}\n"
            "Display the list of user accounts and help the user select one that needs support.\n"
            "### Response Rules:\n"
            "- Present the accounts in a selectable format.\n"
            "- Ask the user to choose the account they need support with.\n"
            "- The response language should match the user's message (Persian or English).\n"
            "- Maintain a polite, friendly, and guiding tone.\n"
            "- Using relevant emojis is allowed.\n"
            "⚠️ Prohibited: Greeting, small talk, or self-introduction.\n"
            f"{OUTPUT_HTML}"
        ),
        agent=ask_witch_account_agent,
        expected_output="Raw HTML text"
    )


def extract_select_account_task(state):
    user_message = state["input"]
    accounts = state["userInfo"]["accounts"]
    select_account = state["userInfo"]["selectAccount"]
    return Task(
        description=(
            f"User message: {user_message}\n"
            f"User accounts: {accounts}\n"
            f"Previously selected account: {select_account}\n\n"
            "Guidelines for account selection:\n"
            "- Determine which account the user intends to choose in the current message.\n"
            "- Direct selection: If the user mentions an exact username, melicode, or any account detail, select that account.\n"
            "- Sequential references:\n"
            "    * Positive order: 'first', 'second', 'third', ... → index = n-1\n"
            "    * Negative order: 'last', 'second to last', 'third from the end', ... → index = -n\n"
            "    * General formula: index = number-1 if positive, index = -number_from_end if negative\n"
            "- Relative references: 'the same as before' → match based on the previously selected account.\n"
            "- Single account: If there is only one account in the list, select it automatically.\n"
            "- Avoid duplicate selections.\n\n"
            "Output instructions:\n"
            "- Return only a list containing exactly **one account** selected by the user.\n"
            "- If multiple accounts are mentioned, select only the first one according to the accounts list.\n"
            "- Extract account information exactly as it appears in the accounts list; do not fabricate or guess values.\n"
            "- If the user's choice is unclear, return an empty list.\n\n"
            "Example of correct JSON output:\n"
            "{\n"
            "  'id': '5383',\n"
            "  'pppoe_username': '989559031030',\n"
            "  'name': 'Saeed',\n"
            "  'lastname': 'Talebi Dowlat Abadi',\n"
            "  'mobile': '09136776910',\n"
            "  'melicode': '6600007117',\n"
            "  'Traffic': '11953',\n"
            "  'service_title': 'One month - 18GB (FD)',\n"
            "  'service_remain_day': 11,\n"
            "  'date_service_start': 1759869000,\n"
            "  'date_service_expire': 1762461000,\n"
            "  'date': 1757310526,\n"
            "  'date_fa': '1404-6-17 09:18',\n"
            "  'date_service_start_fa': '1404-7-16 00:00',\n"
            "  'date_service_expire_fa': '1404-8-16 00:00'\n"
            "}\n\n"
            f"{OUTPUT_HTML}"
        ),
    agent = extract_select_account_agent,
    expected_output = "Only valid JSON"

)