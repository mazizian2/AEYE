from general.state_manager import save_state
import re
from LTE.task.LTETask import create_problem_list_task, get_account_user_task, create_json_account_task, \
    ask_witch_account_task, extract_select_account_task,create_ask_problem_task
import asyncio
from socket_instance import sio
from general.State import ChatState
from general.tools import  run_task_as_crew, create_message,parse_json5
from apis.apis import customer_search



async def handle_ask_problem(state: ChatState) -> ChatState:
    print("handle problem ask")
    user_info = state["userInfo"]['info']
    task = create_ask_problem_task(user_info)
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

async def handle_problem_list(state: ChatState) -> ChatState:
    print("handle problem list")
    user_input = state["input"]
    task = create_problem_list_task(user_input)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    problems = parse_json5(response)
    state["userInfo"]['problems'] = problems
    save_state(state)

    # response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    # message = create_message('assistant', response)
    # state["messages"].append(message)
    # asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if (state['intents'] != []):
        state['intents'].pop()

    if(state['userInfo']['accountInfo']=={}):
        state['next_node'] = 'get_info_account'

    elif(state['userInfo']['selectAccount']==[]):
        state['next_node'] = 'ask_witch_account'

    elif (state['userInfo']['problems'] == []):
        state['next_node'] = "ask_problem"
    else:
        state['next_node'] = ""
    save_state(state)
    return state


async def handle_get_account_user(state: ChatState) -> ChatState:
    print("handle get_account_user")
    task = get_account_user_task(state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if (state['intents'] != []):
        state['intents'].pop()

    state['next_node'] = "extract_json_account"
    save_state(state)
    return state


async def handle_json_account(state: ChatState) -> ChatState:
    print("handle json_account")
    task = create_json_account_task(state)
    result = run_task_as_crew(task)
    response=result.raw.strip()
    info = parse_json5(response)
    state["userInfo"]['accountInfo'] = info
    print("user info json is>>", state["userInfo"]['accountInfo'])
    if (state['intents'] != []):
        state['intents'].pop()

    state['next_node'] = "ask_witch_account"
    save_state(state)
    return state


async def handle_ask_witch_account(state: ChatState) -> ChatState:
    print("handle ask_witch_account")
    result=customer_search(state['userInfo']['accountInfo'],)
    print("state['userInfo']['accountInfo']",state['userInfo']['accountInfo'])
    print("result ['accountInfo']",result['list'])
    accounts=[]
    if(result['result']==True):
        accounts.append(result['list'])
    state["userInfo"]['accounts']=accounts
    print('accounts is>>', accounts)
    save_state(state)
    task = ask_witch_account_task(state)
    result = run_task_as_crew(task)
    response = result.raw.strip()
    response = re.sub(r"^```html\s*|\s*```$", "", response).strip()
    message = create_message('assistant', response)
    state["messages"].append(message)
    asyncio.create_task(sio.emit("room_mehdi", message, room="mehdi"))
    if (state['intents'] != []):
        state['intents'].pop()

    state['next_node'] = "extract_select_account"
    save_state(state)
    return state


async def handle_extract_select_account(state: ChatState) -> ChatState:
    print("handle ask_witch_account")
    task = extract_select_account_task(state)
    result = run_task_as_crew(task)
    response=result.raw.strip()
    select= parse_json5(response)
    state["userInfo"]['selectAccount'] = select
    save_state(state)

    if (state['intents'] != []):
        state['intents'].pop()
    if(state['userInfo']['problems']==[]):
        state['next_node'] = "ask_problem"
    else:
        state['next_node'] = "unknown"
    save_state(state)
    return state
