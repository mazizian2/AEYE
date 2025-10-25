from fastapi import FastAPI,Request
import socketio
from pydantic import BaseModel
from general.tools import (extract_min_max,filter_by_date,clean_and_load_json,execute_stored_procedure,create_message,load_latest_state)
from graph.SHGraph import build_graph
from socket_instance import sio
from general.State import ChatState
from general.state_manager import save_state, load_latest_state
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


app = FastAPI(title="پشتیبانی اتوماسیون اداری")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
class Message(BaseModel):
    content: str
views = Jinja2Templates(directory="views")
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return views.TemplateResponse("index.html", {"request": request, "name": "کاربر"})


@app.post("/langchain-sample4/")
async def process_message(request: Request,message: Message):
    print("coll sample4",message)
    # print(pyodbc.drivers())
    # server = '185.237.85.3'
    # database = 'ICA_DatacenterNew'
    # username = 'sa'
    # password = 'data3755'

    # بدون پارامتر
    rows = execute_stored_procedure( 'B_SefareshList_Sel',[0,0])
    # json_result = execute_stored_procedure(server, database, username, password, 'Robo_User',[3132685653])


    # print(json_result)
    # cm = ChromaManager("user_messages3")
    #
    # print("create chroma")
    # # ثبت پیام‌ها
    # cm.add_message("سلام! می‌خوام یه سفر برنامه‌ریزی کنم.", metadata={"user_id": 1, "topic": "travel"})
    #
    # print("create message")
    # cm.add_message("بودجه حدود ۲۰ میلیون تومنه.", metadata={"user_id": 1, "topic": "budget"})
    #
    # print("create message")
    #
    # # واکشی پیام‌ها
    # results = cm.fetch_similar("بودجه")
    # print("پیام‌های واکشی شده:", results)
    # user_info = json.loads('{"name": "مهدی دهدار", "phone": "09134226929"} ')
    #
    # user = json.loads(user_info)
    # return {"response":user}
    # if os.path.exists(f"states/test.json"):
    #     with open(f"states/test.json", "r", encoding="utf-8") as f:
    #         state_data = json.load(f)
    # print("json:",state_data)
    # latest_state = load_latest_state()
    # graph=create_sh_graph()
    # graph_png = graph.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(graph_png)
    # state: ChatState = dict(latest_state)
    # state = ChatState(
    #     input=message.content,
    #     messages=[],
    #     intent=""
    # )

    # state['input']=message.content
    # state["messages"].append({
    #     "id": str(uuid.uuid4()),
    #     "role": 'user',
    #     "content": message.content,
    #     "timestamp": datetime.now().isoformat()
    # })
    # sio = request.app.state.sio
    # await sio.emit("room_mehdi", {
    #     "id": str(uuid.uuid4()),
    #     "role": 'user',
    #     "content": message.content,
    #     "timestamp": datetime.now().isoformat()
    # }, room="mehdi")
    # print("state:",state)
    # result = graph.invoke(state)
    json_rows=clean_and_load_json(rows)
    filter_row_by_date=filter_by_date(json_rows)
    extract=extract_min_max(filter_row_by_date)
    print("row:",len(filter_row_by_date))
    print("row:",filter_row_by_date)
    print("row:",extract)
    return {"response":filter_row_by_date}

# اتصال کاربران
@sio.event
async def connect(sid, environ):
    print(f"🔌 Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

# جوین شدن به Room
@sio.event
async def join(sid, data):
    print("join",data)
    room = data["room"]
    await sio.enter_room(sid, room)
    await sio.emit("message", f"🔔 A new user joined {room}", room=room)

# دریافت پیام
# @sio.event
# async def message(sid, data):
#     print("message",data)
#     room = data["room"]
#     msg = data["msg"]
#     message=make_message('user',msg)
#     await sio.emit("room_mehdi", message, room="mehdi")
#     latest_state = load_latest_state()
#     graph=create_sh_graph()
#     state: ChatState = dict(latest_state)
#     state['input']=msg
#     state["messages"].append(message)
#     print("state:",state)
#     result = await graph.ainvoke(state)
#     print("print",result)
@sio.event
async def message(sid, data):
    print("message", data)
    room = data["room"]
    msg = data["msg"]
    latest_state = load_latest_state()
    print(latest_state)
    state: ChatState = dict(latest_state)
    message=create_message("user",msg)
    await sio.emit("room_mehdi", message, room="mehdi")
    state["messages"].append(message)
    state['input']=msg
    save_state(state)
    graph = build_graph()
    result = await graph.ainvoke(state)
    print("result:",result)