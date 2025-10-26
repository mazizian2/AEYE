from langgraph.graph import StateGraph, END
from shabak.LTE.general.tools import clean_and_load_json, execute_stored_procedure, make_message, extract_json_from_text, run_task_as_crew, save_state
from typing import TypedDict, Optional, List, Dict, Any
import traceback
import re
from rapidfuzz import fuzz
import json
from shabak.LTE.task.LTETask import create_support_switch_task, LTE_detect_task, return_operator_task, edit_setting_modem_task, \
    setting_modem_task, turn_on_modem_task, check_modem_task, \
    unknown_task, create_detect_user_info_task, create_display_user_info_task, \
    help_internet_task, \
    create_greeting_response_task, create_context_switch_task, \
    detect_select_account_task, ask_witch_user_task, request_collect_user_info_task, \
    collect_user_info_task
import asyncio
import os

from shabak.LTE.general.socket_instance import sio

INTENTS = [
    "support",
    "buy_service",
    "unknown",
    "greeting",
    "request_collect_user_info",


]
SUBINTENTS = [
    "help_internet",
    "detect_user_info",
    "collect_user_info",
    "ask_which_account",
    "detect_select_account",
    "check_modem",
    "turn_on_modem",
    "setting_modem",
    "edit_setting_modem",
    "return_operator",
]

from langchain_openai import ChatOpenAI
from openai import OpenAI


class Message(TypedDict):
    role: str  # "user" یا "assistant"
    content: str


class ChatState(TypedDict, total=False):
    input: str
    status: Optional[str]
    user_info_list: List[Dict[str, str]]
    intent: List[str]
    sub_intent: List[str]
    history_intent: List[str]
    history_sub_intent: List[str]
    messages: List[Message]
    context_stack: List[str]
    current_intent: Optional[str]
    selected_account: Optional[Dict[str, str]]
    user_info_completed: Optional[bool]
    lte_status: Optional[Dict[str, str]]
    next_node: Optional[str]
    awaiting_account_choice: Optional[bool]
    # service_filters: Optional[Dict[str, Any]]


def get_last_intent(state: dict) -> str:
    """
    آخرین intent ذخیره‌شده را برمی‌گرداند.
    - اگر لیست intent خالی بود یا وجود نداشت → رشته خالی برگردانده می‌شود.
    """
    intents = state.get("intent", [])
    if intents:
        return intents[-1]
    return ""
def get_last_sub_intent(state: dict) -> str:
    """
    آخرین intent ذخیره‌شده را برمی‌گرداند.
    - اگر لیست intent خالی بود یا وجود نداشت → رشته خالی برگردانده می‌شود.
    """
    intents = state.get("sub_intent", [])
    if intents:
        return intents[-1]
    return ""


def update_intent(state: Dict[str, Any], new_intents: List[str]) -> Dict[str, Any]:
    """
    آیتم‌های intent جدید را به صورت تکی در state['intent'] ذخیره می‌کند.
    - اگر دقیقا مشابه آخرین ساب‌سیکوئنس ذخیره‌شده باشد → اضافه نمی‌شود.
    - فقط 10 مورد آخر نگه داشته می‌شود.
    """
    print("update intent>>", state)
    if "intent" not in state or state["intent"] is None:
        state["intent"] = []

    history = state["intent"]

    # اگر new_intents پشت سر هم دقیقا تکرار شده باشند → رد کن
    if len(history) >= len(new_intents) and history[-len(new_intents):] == new_intents:
        return state

    # در غیر این صورت تک تک اضافه کن
    for item in new_intents:
        history.append(item)

        # حداکثر 10 مورد آخر
        if len(history) > 10:
            history[:] = history[-10:]

    state['intent'] = history
    return state

def update_sub_intent(state: Dict[str, Any], new_intents: List[str]) -> Dict[str, Any]:
    """
    آیتم‌های intent جدید را به صورت تکی در state['intent'] ذخیره می‌کند.
    - اگر دقیقا مشابه آخرین ساب‌سیکوئنس ذخیره‌شده باشد → اضافه نمی‌شود.
    - فقط 10 مورد آخر نگه داشته می‌شود.
    """
    print("update sub intent>>", state)
    if "sub_intent" not in state or state["sub_intent"] is None:
        state["sub_intent"] = []

    history = state["sub_intent"]

    # اگر new_intents پشت سر هم دقیقا تکرار شده باشند → رد کن
    if len(history) >= len(new_intents) and history[-len(new_intents):] == new_intents:
        return state

    # در غیر این صورت تک تک اضافه کن
    for item in new_intents:
        history.append(item)

        # حداکثر 10 مورد آخر
        if len(history) > 10:
            history[:] = history[-10:]

    state['sub_intent'] = history
    return state


def update_history_intent(state: Dict[str, Any], new_intent: str) -> None:
    """
    وضعیت intent جدید را در state['history_intent'] ذخیره می‌کند.
    - اگر مشابه آخرین intent باشد → اضافه نمی‌شود.
    - حداکثر 10 مورد نگه داشته می‌شود.
    """
    if "history_intent" not in state or state["history_intent"] is None:
        state["history_intent"] = []

    history = state["history_intent"]

    # اگر آخرین مورد تکراریه → نادیده بگیر
    if history and history[-1] == new_intent:
        return state

    # اضافه کردن intent جدید
    history.append(new_intent)

    # اگر بیشتر از 10 تا شد → فقط 10 تای آخر بمونه
    if len(history) > 10:
        history[:] = history[-10:]

    state['history_intent'] = history
    return state
def update_history_sub_intent(state: Dict[str, Any], new_intent: str) -> None:
    """
    وضعیت intent جدید را در state['history_intent'] ذخیره می‌کند.
    - اگر مشابه آخرین intent باشد → اضافه نمی‌شود.
    - حداکثر 10 مورد نگه داشته می‌شود.
    """
    if "history_sub_intent" not in state or state["history_sub_intent"] is None:
        state["history_sub_intent"] = []

    history = state["history_sub_intent"]

    # اگر آخرین مورد تکراریه → نادیده بگیر
    if history and history[-1] == new_intent:
        return state

    # اضافه کردن intent جدید
    history.append(new_intent)

    # اگر بیشتر از 10 تا شد → فقط 10 تای آخر بمونه
    if len(history) > 10:
        history[:] = history[-10:]

    state['history_sub_intent'] = history
    return state


def handle_start(state: ChatState):
    print(">>> handleStart", state)

    state["intent"] = state.get("intent", [])
    state["history_intent"] = state.get("history_intent", [])

    next_node = state["next_node"]
    user_input = state["input"]
    last_ai_message = None
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            last_ai_message = msg.get("content")
            break
    task = create_context_switch_task(user_input, state, last_ai_message)
    result = run_task_as_crew(task)
    intent = result.raw.strip()
    print("handle start>>>>", result)
    if intent not in INTENTS:
        state = update_intent(state, ["unknown"])
        state = update_history_intent(state, "unknown")

    if next_node != intent:
        state = update_intent(state, [intent])
        print("handle start>>", state)
        state = update_history_intent(state, intent)

    print("handle start>>>", state)
    state = update_intent(state, ["detect_status"])

    save_state(state)

    return state

def handle_sub_intent(state: ChatState):
    print(">>> handleStart", state)

    state["sub_intent"] = state.get("sub_intent", [])
    state["history_sub_intent"] = state.get("history_sub_intent", [])

    next_node = state["next_node"]
    user_input = state["input"]
    last_ai_message = None
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            last_ai_message = msg.get("content")
            break
    task = create_support_switch_task(user_input, state, last_ai_message)
    result = run_task_as_crew(task)
    intent = result.raw.strip()
    print("handle start>>>>", result)
    if intent not in SUBINTENTS:
        state = update_sub_intent(state, ["unknown"])
        state = update_history_sub_intent(state, "unknown")

    if next_node != intent:
        state = update_sub_intent(state, [intent])
        print("handle start>>", state)
        state = update_history_sub_intent(state, intent)

    print("handle start>>>", state)
    state = update_sub_intent(state, ["detect_status"])

    save_state(state)

    return state

def detect_intent(state: ChatState):
    print(">>>> Detect Status User",state["intent"])
    # print(state["intent"])
    if not state.get("user_info_completed", False):
        # اگر اطلاعات کامل نیست، مستقیم میره سراغ نود گرفتن اطلاعات
        state['next_node'] = "detect_user_info"
        return state
    # بررسی وجود intent
    if "intent" not in state or not isinstance(state["intent"], list):
        state["intent"] = []

    elif state["intent"]:
        state["intent"].pop()
    print(">>>> Detect Status User2",state["intent"])

    intent = get_last_intent(state)
    if not intent:
        print("⚠️ No intent found, defaulting to 'unknown'")
        intent = "unknown"

    if intent == 'support':
        state=update_intent(state,["detect_user_info"])
    elif intent == 'detect_user_info':
        state['intent'].pop()
        state=update_intent(state,["support","detect_user_info","handle_sub_intent"])
        state=update_history_intent(state,"support")
        print(">>>> Detect Status User3", state["intent"])
    elif intent == 'support_other_account':
        state['intent'].pop()
        state=update_history_intent(state,"support")
        state["selected_account"] = {}
        state=update_intent(state,["support","detect_user_info"])
    elif intent == 'buy_service_other_account':
        state['intent'].pop()
        state["selected_account"] = {}
        state=update_history_intent(state,"buy_service")
        state=update_intent(state,["buy_service","detect_user_info"])
    elif intent == 'request_collect_user_info':
        state = update_history_intent(state, "collect_user_info")
        state = update_intent(state, ["collect_user_info"])

    state['next_node'] = get_last_intent(state)
    save_state(state)
    return state

def detect_sub_intent(state: ChatState):
    print(">>>> Detect sub Status User>>>")
    print(state["sub_intent"])

    # بررسی وجود intent
    if "sub_intent" not in state or not isinstance(state["sub_intent"], list):
        state["sub_intent"] = []

    elif state["sub_intent"]:
        state["sub_intent"].pop()

    intent = get_last_intent(state)
    if not intent:
        print("⚠️ No intent found, defaulting to 'unknown'")
        intent = "unknown"

    # مسیرهای مودم
    if intent == "check_modem":
        state = update_sub_intent(state, ["turn_on_modem"])
    elif intent == "turn_on_modem":
        state = update_sub_intent(state, ["setting_modem"])
    elif intent == "setting_modem":
        state = update_sub_intent(state, ["edit_setting_modem"])
    elif intent == "edit_setting_modem":
        state = update_sub_intent(state, ["return_operator"])
    elif intent == "return_operator":
        print("Modem setup flow completed.")

    # تعیین نود بعدی
    state["next_node"] = get_last_sub_intent(state) or "unknown"
    save_state(state)
    return state


# def detect_status(state: ChatState):
#     print(">>>>detect Status User")
#     state['intent'].pop()
#     intent = get_last_intent(state)
#     # if intent == 'support':
#     #     state=update_intent(state,["detect_user_info"])
#     # elif intent == 'detect_user_info':
#     #     state['intent'].pop()
#     #     state=update_intent(state,["support","detect_user_info"])
#     #     state=update_history_intent(state,"support")
#     #     state["selected_account"] = {}
#     # elif intent == 'support_other_account':
#     #     state['intent'].pop()
#     #     state=update_history_intent(state,"support")
#     #     state["selected_account"] = {}
#     #     state=update_intent(state,["support","detect_user_info"])
#     # elif intent == 'buy_service_other_account':
#     #     state['intent'].pop()
#     #     state["selected_account"] = {}
#     #     state=update_history_intent(state,"buy_service")
#     #     state=update_intent(state,["buy_service","detect_user_info"])
#
#     # === مسیرهای مربوط به مودم ===
#     if intent == 'check_modem':
#         state = update_intent(state, ["turn_on_modem"])
#
#     elif intent == 'turn_on_modem':
#         state = update_intent(state, ["setting_modem"])
#
#     elif intent == 'setting_modem':
#         state = update_intent(state, ["edit_setting_modem"])
#
#     elif intent == 'edit_setting_modem':
#         state = update_intent(state, ["return_operator"])
#
#     elif intent == 'return_operator':
#         print("Modem setup flow completed.")
#     state['next_node'] = get_last_intent(state)
#     save_state(state)
#     return state

async def detect_user_info(state: ChatState):
    print(">>>>detect User Info")
    state['intent'].pop()

    user_accounts = state.get("user_info_list", [])
    user_message = state.get("input", "")

    # اگر قبلا اکانت انتخاب شده باشه → مستقیم ادامه بده
    selected_account = state.get("selected_account", "")
    print("detect user info:", selected_account)
    if selected_account:
        print("اکانت قبلاً انتخاب شده:", selected_account["UserName"])
        state["next_node"] = get_last_intent(state)
        save_state(state)
        print("save_state:", state['intent'])

        return state

    # صدا زدن تسک
    task = create_detect_user_info_task(user_message, user_accounts)
    result = run_task_as_crew(task)
    response = result.raw.strip()

    try:
        print("json value", response)
        decision = clean_and_load_json(response)
        intent = decision.get("intent")
        action = decision.get("action")
        username = decision.get("username")
        message = decision.get("message")

        print("1")
        if intent == "switch_account":
            if action == "select":

                state = update_user_info_from_response(response, state, username)
                state["next_node"] = get_last_intent(state)
                # پیدا شده توی لیست → انتخاب کن
                # state["selected_account"] = next(acc for acc in user_accounts if acc["UserName"] == username)
                # state["next_node"] = state['intent'].pop()

            elif action == "ask_select":

                state = update_intent(state, ["ask_which_account"])
                state = update_history_intent(state, "ask_which_account")
                state['next_node'] = "ask_which_account"
                # response = re.sub(r"^```html\s*|\s*```$", "", message).strip()
                #
                # message = make_message('assistant', response)
                # asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))

                # باید از کاربر بخوایم انتخاب کنه
                # state["next_node"] = "ask_which_account"
                # state["messages"].append(make_message("assistant", message))

            elif action == "new_username":
                print("3")
                # state = update_user_info_from_response(response, state, username)
                state = update_intent(state, ["request_collect_user_info"])
                state = update_history_intent(state, "request_collect_user_info")
                state["next_node"] ="request_collect_user_info"
                print("state:", state)
                # کد جدید → مستقیم بره برای ذخیره‌سازی
                # state["next_node"] = "collect_user_info"
                # state["selected_account"] = {"UserName": username} if username else None

            elif action == "keep_current":
                state["next_node"] = get_last_intent(state)
            else:
                print("⚠️ خروجی نامعتبر از ایجنت:", decision)
        else:
            if action == "keep_current":
                state["next_node"] = get_last_intent(state)

    except Exception as e:
        print("❌ خطا در پردازش detect_user_info:", e)
        traceback.print_exc()

    save_state(state)
    return state


def request_collect_user_info(state: ChatState):
    print("در حال جمع‌آوری اطلاعات کامل برای کاربر جدید...")
    task = request_collect_user_info_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    state['intent'].pop()
    state = update_intent(state, ["collect_user_info"])
    state = update_history_intent(state, "collect_user_info")
    state['next_node'] = "collect_user_info"
    message = make_message('assistant', response)
    state["messages"].append(message)
    save_state(state)
    print("request collect user info", response)

    # asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    return state


def update_user_info_from_response(response: str, state: ChatState, new_username="") -> dict:
    """
    این تابع خروجی تسک (response) و state فعلی رو می‌گیره
    و بر اساس یوزرنیم استخراج شده، اطلاعات کاربر رو به‌روز می‌کنه.

    پارامترها:
        response (str): پاسخ ایجنت (معمولاً شامل JSON)
        state (dict): وضعیت جاری چت

    خروجی:
        dict: state به‌روزرسانی شده
    """
    try:
        # استخراج JSON
        if new_username == "":
            user_info = extract_json_from_text(response)
            new_username = user_info.get("username", "").strip()

        if not new_username:
            raise ValueError("❌ هیچ username معتبری در پاسخ پیدا نشد.")

        updated = False
        for i, u in enumerate(state.get("user_info_list", [])):
            existing_username = u.get("username", "").strip()

            # ۱. اگر دقیقاً یکی بود → نادیده گرفتن
            if new_username == existing_username:
                print(f"ℹ️ کد {new_username} تکراری است، نیازی به ذخیره نیست.")
                updated = True
                break

            # ۲. اگر خیلی شبیه بود (مثل اضافه بودن صفر در اول)
            similarity = fuzz.ratio(existing_username, new_username)
            print("Similarity>>", similarity)
            if similarity >= 95:
                print(f"♻️ کد مشابه پیدا شد ({similarity}%)، از قبلی استفاده می‌کنیم.")
                new_username = existing_username
                updated = True
                break

        # اگر یوزرنیم جدید بود → به لیست اضافه کن
        if not updated:
            state.setdefault("user_info_list", []).append({"username": new_username})
            print(f"✅ کد جدید {new_username} اضافه شد.")

        # گرفتن اطلاعات کامل از دیتابیس
        full_user_info = execute_stored_procedure('Robo_User', [new_username])
        try:
            data_list = json.loads(full_user_info)  # تبدیل رشته به لیست از دیکشنری‌ها

            if data_list:
                # فقط عنصر اول لیست رو انتخاب کن
                state["selected_account"] = data_list[0]
            else:
                # اگه لیست خالی بود
                state["selected_account"] = {"username": new_username}

        except json.JSONDecodeError:
            print("❌ خطا در تبدیل JSON از SP")
            state["selected_account"] = {"username": new_username}

    except Exception as e:
        print("⚠️ خطا در update_user_info_from_response:", e)
        state["selected_account"] = {"username": None}

    return state


def collect_user_info(state: ChatState):
    print("در حال جمع‌آوری کد کاربری (username) برای کاربر جدید...")

    task = collect_user_info_task(state['input'])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    # پیام هوش مصنوعی رو ذخیره کنیم
    message = make_message('assistant', response)
    state["messages"].append(message)
    state = update_user_info_from_response(response, state)

    save_state(state)
    return state


async def ask_which_account(state: ChatState):
    print("روی کدوم کاربر")
    accounts_str = "\n".join(
        [f"{i + 1}. {acc['username']}" for i, acc in enumerate(state['user_info_list'])]
    )
    task = ask_witch_user_task(accounts_str)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intent'].pop()
    state = update_intent(state, ["detect_select_account"])
    state = update_history_intent(state, "detect_select_account")
    state['next_node'] = "detect_select_account"
    save_state(state)
    return state


async def handle_greeting(state: ChatState):
    print("سلام")
    task = create_greeting_response_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    state['intent'].pop()
    state['next_node'] = ""
    save_state(state)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    return state


async def help_internet(state: ChatState):
    print("سلام")
    task = help_internet_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intent'].pop()
    save_state(state)
    return state


async def help_internet(state: ChatState):
    print("سلام")
    task = help_internet_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intent'].pop()
    save_state(state)
    return state


llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#
# async def create_image_rangobarg(state: ChatState):
#     print("ایجاد عکس")
#     task = create_image_prompt_task(state["input"])
#     print(task)
#     result = run_task_as_crew(task)
#     respons = result.raw.strip()
#     response = llm.images.generate(
#         model="gpt-image-1",
#         prompt=respons,
#         size="1024x1024"
#     )
#
#     image_url = response.data[0].url
#     print('image_url is>>')
#     print(image_url)
#     state["messages"].append(image_url)
#     asyncio.create_task(sio.emit("room_mehdi", image_url, room="mehdi"))
#     save_state(state)
#     return state


def detect_select_account(state: ChatState):
    print("انتخاب کاربر")
    state['intent'].pop()
    accounts_str = "\n".join(
        [f"{i + 1}. {acc['username']}" for i, acc in enumerate(state['user_info_list'])]
    )
    task = detect_select_account_task(state['input'], accounts_str)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    print("response detect select account:", response)
    match = re.search(r"\d+", response)
    print("match:", match)
    if match:
        index = int(match.group(0)) - 1
        if 0 <= index < len(state['user_info_list']):
            state["selected_account"] = state['user_info_list'][index]
            state = update_user_info_from_response("", state, state["selected_account"]['username'])

    state['next_node'] = get_last_intent(state)
    save_state(state)
    return state


async def handle_support(state: ChatState):
    print(">>> handle support >>>", state)
    task = create_display_user_info_task(state["selected_account"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    state['intent'].pop()
    state['next_node'] = ""
    save_state(state)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    return state


# async def handle_buy_service(state: ChatState):
#     print(">>> handle buy service >>>", state)
#     service_filters = state.get("service_filters", [])
#     rows = execute_stored_procedure('B_SefareshList_Sel', [0, 0])
#
#     json_rows = clean_and_load_json(rows)
#     filter_row_by_date = filter_by_date(json_rows)
#     extract = extract_min_max(filter_row_by_date)
#
#     task = detect_buy_service_task(state["input"], service_filters, extract)
#     result = run_task_as_crew(task)
#     response = result.raw.strip()
#     data = clean_and_load_json(response)
#     state['service_filters'] = data
#     message = make_message('assistant', data)
#     state["messages"].append(message)
#     state['intent'].pop()
#     state['next_node'] = ""
#     save_state(state)
#
#     if data["user_intent"]["complete_query"]:
#         result = apply_filters(json_rows, data, 5)
#         print("result", result)
#         print("🔎 کاربر تایید کرده، بر اساس همین داده‌ها نتیجه را نمایش بده.")
#         task = generate_service_list_task(result)
#         result = run_task_as_crew(task)
#         response = result.raw.strip()
#         response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
#         message = make_message('assistant', response)
#         state["messages"].append(message)
#         save_state(state)
#         asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
#     else:
#         print("ℹ️ هنوز کاربر تایید نکرده، باید سوالات تکمیلی پرسیده شود.")
#         task = clarify_filters_task(state["selected_account"], data)
#         result = run_task_as_crew(task)
#         response = result.raw.strip()
#         response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
#         message = make_message('assistant', response)
#         state["messages"].append(message)
#         save_state(state)
#         asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
#
#     return state


async def handle_LTE_detect(state: ChatState):
    print(">>>handle LTE detect >>>", state)
    task = LTE_detect_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    data = clean_and_load_json(response)
    state['lte_status'] = data
    message = make_message('assistant', data)
    state["messages"].append(message)
    state['intent'].pop()
    state['next_node'] = ""
    save_state(state)
    return state


async def handle_unknown(state: ChatState):
    print(">>> handle unknown >>>", state)
    print("سلام")
    task = unknown_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    state['intent'].pop()
    state['next_node'] = ""
    save_state(state)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    return state


async def check_modem(state: ChatState):
    print("check_modem")
    task = check_modem_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intent'].pop()
    save_state(state)
    return state


async def turn_on_modem(state: ChatState):
    print("turn_on_modem")
    task = turn_on_modem_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intent'].pop()
    save_state(state)
    return state


async def setting_modem(state: ChatState):
    print("setting_modem")
    task = setting_modem_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intent'].pop()
    save_state(state)
    return state


async def edit_setting_modem(state: ChatState):
    print("check_modem")
    task = edit_setting_modem_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intent'].pop()
    save_state(state)
    return state


async def return_operator(state: ChatState):
    print("return_operator")
    task = return_operator_task(state["input"])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = make_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intent'].pop()
    save_state(state)
    return state


def create_sh_graph():
    builder2 = StateGraph(ChatState)
    builder2.add_node("start", handle_start)
    builder2.add_node("handle_LTE_detect", handle_LTE_detect)
    builder2.add_node("detect_status", detect_intent)
    builder2.add_node("detect_sub_status", detect_sub_intent)
    builder2.add_node("check_modem", check_modem)
    builder2.add_node("turn_on_modem", turn_on_modem)
    builder2.add_node("setting_modem", setting_modem)
    builder2.add_node("edit_setting_modem", edit_setting_modem)
    builder2.add_node("return_operator", return_operator)
    builder2.add_node("unknown", handle_unknown)
    builder2.add_node("greeting", handle_greeting)
    builder2.add_node("collect_user_info", collect_user_info)
    builder2.add_node("detect_user_info", detect_user_info)
    builder2.add_node("help_internet", help_internet)
    builder2.add_node("detect_select_account", detect_select_account)
    builder2.add_node("support", handle_support)
    builder2.add_node("handle_sub_intent", handle_sub_intent)
    builder2.add_node("request_collect_user_info", request_collect_user_info)

    builder2.set_entry_point("start")

    builder2.add_edge("start", "detect_status")
    # builder2.add_edge("detect_status", "request_collect_user_info")
    # builder2.add_edge("detect_sub_status", "handle_LTE_detect")

    builder2.add_conditional_edges(
        "detect_status",
        lambda state: state["next_node"],
        {
            "greeting": "greeting",
            "help_internet": "help_internet",
            "collect_user_info": "collect_user_info",
            "detect_user_info": "detect_user_info",
            "handle_sub_intent": "handle_sub_intent",
            "request_collect_user_info":"request_collect_user_info",
            # "ask_which_account": "ask_which_account",
            "detect_select_account": "detect_select_account",
            "support": "support",
            # "buy_service": "buy_service",
            "unknown": "unknown",
            "handle_LTE_detect":"handle_LTE_detect",
            "check_modem": "check_modem",
            "turn_on_modem": "turn_on_modem",
            "setting_modem": "setting_modem",
            "edit_setting_modem": "edit_setting_modem",
            "return_operator": "return_operator",

        }
    )

    # builder2.add_conditional_edges(
    #     "detect_sub_status",
    #     lambda state: state["next_node"],
    #     {
    #         "unknown": "unknown",
    #
    #         "check_modem": "check_modem",
    #         "turn_on_modem": "turn_on_modem",
    #         "setting_modem": "setting_modem",
    #         "edit_setting_modem": "edit_setting_modem",
    #         "return_operator": "return_operator",
    #
    #     }
    # )

    builder2.add_edge("unknown", END)
    builder2.add_edge("return_operator", END)
    builder2.add_edge("edit_setting_modem", END)
    builder2.add_edge("turn_on_modem", END)

    graph2 = builder2.compile()

    return graph2
