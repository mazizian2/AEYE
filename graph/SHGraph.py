from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from task.SHTask import create_service_suggestion_response_task, create_json_need_buy_response_task, \
    create_express_need_buy_response_task, create_greeting_response_task, context_switch_task, register_order_json_task, \
    create_objection_buy_response_task, user_info_collector_response_task, register_order_response_task, \
    user_info_json_task, unknown_response_task, create_answer_service_suggestion_response_task, \
    create_service_suggestion_json_task, payment_task
from LTE.graph.LTEGraph import handle_problem_list, handle_get_account_user, handle_json_account, \
    handle_ask_witch_account, handle_extract_select_account, handle_ask_problem

from general.state_manager import save_state, load_latest_state
from general.State import ChatState, ChatStateManager
from general.tools import run_async_task_as_crew, run_task_as_crew, create_message, parse_json5, getIntent, \
    get_last_intent
import asyncio
from socket_instance import sio
import re
import json

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
    "get_info_account",
    "extract_json_account",
    "ask_witch_account",
    "extract_select_account",
    "unknown"
]


def analyze_message(state: ChatState) -> ChatState:
    user_input = state["input"]
    last_ai_message = None
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            last_ai_message = msg.get("content")
            break
    assistant_messages = [m["content"] for m in state.get("messages", []) if m.get("role") == "assistant"]

    task = context_switch_task(user_input, state, last_ai_message, assistant_messages)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    print("response intent is", response)
    state["intents"].append(response)
    state["next_node"] = response
    save_state(state)
    return state


def detect_intent(state: ChatState):
    print("detect_intent")

    intent = get_last_intent(state)
    user_need = state['userInfo']['needs']
    print("intents is >>>", intent)
    # if intent=="extract_json_buy":
    #     if user_need['type']==[] :
    #         state['next_node'] = "express_need_buy"
    if intent == "unknown":
        state['next_node'] = "unknown"
    # elif intent == "register_order":
    #     state['next_node'] = ""
    # elif intent == "service_suggestion_buy":
    #     state['next_node'] = "service_suggestion_buy"

    save_state(state)
    return state


async def unknown(state: ChatState) -> ChatState:
    print("handle unknown")
    user_input = state["input"]

    task = unknown_response_task(user_input)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if (state['intents'] != []):
        state['intents'].pop()

    state['next_node'] = ""
    save_state(state)
    return state


async def handle_greeting(state: ChatState):
    print("handle greeting")
    intent = get_last_intent(state)
    user_input = state["input"]
    user_info = state["userInfo"]
    assistant_messages = [m["content"] for m in state.get("messages", []) if m.get("role") == "assistant"]

    if intent == 'greeting':
        task = create_greeting_response_task(user_input, assistant_messages, user_info)
        result = run_task_as_crew(task)
        response = result.raw.strip()
        response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
        message = create_message('assistant', response)
        state["messages"].append(message)
        asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
        state['intents'].pop()

    state['next_node'] = ""
    save_state(state)
    return state


async def handle_express_need_buy(state: ChatState):
    print("handle express need buy")
    intent = get_last_intent(state)
    user_input = state["input"]
    user_info = state["userInfo"]

    if intent == "express_need_buy":
        task = create_express_need_buy_response_task(user_input, state)
        result = run_task_as_crew(task)
        response = result.raw.strip()
        response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
        message = create_message('assistant', response)
        state["messages"].append(message)
        asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
        state['intents'].pop()
    # if user_info['needs']
    state['next_node'] = ""
    save_state(state)
    return state


def handle_extract_json_buy(state: ChatState):
    intent = get_last_intent(state)
    user_input = state["input"]
    user_info = state["userInfo"]
    user_need = state["userInfo"]["needs"]
    last_ai_message = None
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            last_ai_message = msg.get("content")
            break
    task = create_json_need_buy_response_task(user_input, last_ai_message, state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    # state["messages"].append(message)
    needs = parse_json5(response)
    print('needs json is>>', needs)
    state["userInfo"]['needs'] = needs
    print('needs is>>', needs)
    print('needs type is>>', needs['type'])
    print('needs type len is>>', len(needs['type']))
    state['intents'].pop()
    if len(needs['type']) == 0:
        state['next_node'] = "express_need_buy"
    elif len(needs['type']) != 0:
        state['next_node'] = "service_suggestion_buy"
    save_state(state)
    return state


async def handle_service_suggestion_buy(state: ChatState):
    intent = get_last_intent(state)
    user_input = state["input"]
    user_info = state["userInfo"]
    print('user needs >>', user_info['needs'])
    print("handle service suggestion buy")

    task = create_service_suggestion_response_task(user_input, user_info, user_info['needs'])
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    # state['intents'].pop()

    state['next_node'] = "set_history_suggestion"
    save_state(state)
    return state


def handle_service_suggestion_json(state: ChatState):
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            last_ai_message = msg.get("content")
            break
    task = create_service_suggestion_json_task(last_ai_message)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    # message = create_message('assistant', response)
    # state["messages"].append(message)
    service_suggestion = parse_json5(response)
    print('service suggestion json is>>', service_suggestion)
    index = len(state["history_suggestion"])
    state["history_suggestion"].append({index: service_suggestion})
    print('needs is>>', state["history_suggestion"])
    if (state['intents'] != []):
        state['intents'].pop()

    state['next_node'] = ""
    save_state(state)
    return state


async def handle_create_answer_service_suggestion(state: ChatState):
    intent = get_last_intent(state)
    user_input = state["input"]
    user_info = state["userInfo"]
    print('user needs >>', user_info['needs'])
    print("handle answer service suggestion")

    task = create_answer_service_suggestion_response_task(user_input, state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if len(state['intents']) != 0:
        state['intents'].pop()

    state['next_node'] = ""
    save_state(state)
    return state


async def handle_register_order_json_buy(state: ChatState):
    user_input = state["input"]

    task = register_order_json_task(user_input, state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    orders = parse_json5(response)
    print('orders json is>>', orders)
    state["userInfo"]['orders'].extend(orders)
    print('orders is>>', state["userInfo"]['orders'])
    print('orders len is>>', state['intents'])
    if len(state['intents']) != 0:
        state['intents'].pop()
    info = state['userInfo']['info']
    if (info['name'] == "" or info['phone'] == ""):
        state['next_node'] = "user_info_collector"
    else:
        state['next_node'] = "payment"

    save_state(state)
    return state


async def handle_objection_buy(state: ChatState):
    print("handle objection buy")
    intent = get_last_intent(state)
    user_input = state["input"]
    user_info = state["userInfo"]

    task = create_objection_buy_response_task(user_input, user_info)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    state['intents'].pop()

    state['next_node'] = ""
    save_state(state)
    return state


async def handle_user_info_collector(state: ChatState):
    print("handle user info collector")
    user_input = state["input"]
    task = user_info_collector_response_task(user_input, state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if (state['intents'] != []):
        state['intents'].pop()
    state['next_node'] = ""
    save_state(state)
    return state


async def handle_user_info_json(state: ChatState):
    print("handle user info json")
    user_input = state["input"]
    task = user_info_json_task(user_input, state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    # state["messages"].append(message)
    info = parse_json5(response)
    state["userInfo"]['info'] = info
    print("user info json is>>", state["userInfo"]['info'])
    # asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if (state['intents'] != []):
        state['intents'].pop()

    if (state["userInfo"]['info']['name'] != "" and state["userInfo"]['info']['phone'] != ""):
        state['next_node'] = "payment"
    else:
        state['next_node'] = "user_info_collector"
    save_state(state)
    return state


async def handle_payment(state: ChatState):
    user_input = state["input"]
    print("handle payment")
    task = payment_task(user_input, state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if len(state['intents']) != 0:
        state['intents'].pop()

    state['next_node'] = ""
    save_state(state)
    return state


async def handle_follow_up_buy(state: ChatState):
    print("handle follow up buy")
    user_input = state["input"]
    task = register_order_response_task(user_input, state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    # state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if (state['intents'] != []):
        state['intents'].pop()

    state['next_node'] = ""
    save_state(state)
    return state


async def handle_support(state: ChatState):
    print("handle support")
    userInfo = state["userInfo"]
    if (userInfo['accounts'] == []):
        state['next_node'] = "get_info_account"
    elif (userInfo['selectAccount'] == []):
        state['next_node'] = "ask_witch_account"
    elif (userInfo['problems'] == []):
        state['next_node'] = "ask_problem"
    else:
        state['next_node'] = "unknown"
    save_state(state)
    return state


# 🔹 ساخت گراف
def build_graph():
    builder = StateGraph(ChatState)
    # نود تحلیل پیام
    builder.add_node("analyze_message", analyze_message)
    builder.add_node("detect_intent", detect_intent)
    builder.add_node("greeting", handle_greeting)
    builder.add_node("express_need_buy", handle_express_need_buy)
    builder.add_node("extract_json_buy", handle_extract_json_buy)
    builder.add_node("service_suggestion_buy", handle_service_suggestion_buy)
    builder.add_node("set_history_suggestion", handle_service_suggestion_json)
    builder.add_node("answer_question_service_suggestion", handle_create_answer_service_suggestion)
    builder.add_node("register_order", handle_register_order_json_buy)
    # builder.add_node("objection_buy", handle_objection_buy)
    builder.add_node("user_info_collector", handle_user_info_collector)
    builder.add_node("extract_user_info_json", handle_user_info_json)
    builder.add_node("follow_up_buy", handle_follow_up_buy)
    builder.add_node("unknown", unknown)
    builder.add_node("payment", handle_payment)

    # support node
    builder.add_node("support", handle_support)
    builder.add_node("set_problem", handle_problem_list)
    builder.add_node("ask_problem", handle_ask_problem)
    builder.add_node("get_info_account", handle_get_account_user)
    builder.add_node("extract_json_account", handle_json_account)
    builder.add_node("ask_witch_account", handle_ask_witch_account)
    builder.add_node("extract_select_account", handle_extract_select_account)
    builder.add_edge("extract_json_account", "ask_witch_account")
    builder.add_conditional_edges("extract_select_account", lambda state: state["next_node"], {
        "ask_problem": "ask_problem",
        "unknown": "unknown"
    }),
    builder.add_conditional_edges("support", lambda state: state["next_node"], {
         "ask_problem": "ask_problem",
         "set_problem": "set_problem",
         "get_info_account": "get_info_account",
         "extract_json_account": "extract_json_account",
         "ask_witch_account": "ask_witch_account",
         "extract_select_account": "extract_select_account",
         "unknown": "unknown"
     }),
    builder.add_conditional_edges("set_problem", lambda state: state["next_node"], {
          "support": "support",
          "ask_witch_account": "ask_witch_account",
          "get_info_account": "get_info_account",
          "unknown": "unknown"
      }),

    builder.set_entry_point("analyze_message")

    builder.add_edge("analyze_message", "detect_intent")
    builder.add_edge("extract_json_buy", "service_suggestion_buy")
    builder.add_edge("service_suggestion_buy", "set_history_suggestion")
    builder.add_conditional_edges("extract_user_info_json", lambda state: state["next_node"], {
        "user_info_collector": "user_info_collector",
        "payment": "payment",
        "follow_up_buy": "follow_up_buy",
        "unknown": "unknown"
    }),

    builder.add_conditional_edges("register_order", lambda state: state["next_node"], {
        "user_info_collector": "user_info_collector",
        "payment": "payment",
        "follow_up_buy": "follow_up_buy",
        "unknown": "unknown"
    }),

    builder.add_conditional_edges("detect_intent", lambda state: state["next_node"], {
        "greeting": "greeting",
        "express_need_buy": "express_need_buy",
        "extract_json_buy": "extract_json_buy",
        "answer_question_service_suggestion": "answer_question_service_suggestion",
        "register_order": "register_order",
        "payment": "payment",
        # "objection_buy": "objection_buy",
        "user_info_collector": "user_info_collector",
        "extract_user_info_json": "extract_user_info_json",
        "follow_up_buy": "follow_up_buy",
        "support": "support",
        "set_problem": "set_problem",
        "extract_json_account": "extract_json_account",
        "extract_select_account": "extract_select_account",
        "unknown": "unknown"
    })

    builder.add_edge("set_history_suggestion", END),
    builder.add_edge("unknown", END),
    builder.add_edge("greeting", END),
    builder.add_edge("user_info_collector", END),
    builder.add_edge("follow_up_buy", END),
        # builder.add_edge("objection_buy", END),
    builder.add_edge("express_need_buy", END),

    return builder.compile()
