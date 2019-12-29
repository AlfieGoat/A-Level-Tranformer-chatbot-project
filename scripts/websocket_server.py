import websockets
import asyncio
import training_skeleton
import json

ts = training_skeleton.TrainingSkeleton()
ts.load_model("0_17750")


async def get_msg_and_reply(websocket):
    conversation = await websocket.recv()  # Waits for a message
    conversation = json.loads(conversation)  # Loads the JSON
    enc = ""  # Create the encoder string
    # Iterates over the user and bot arrays and adds them to enc + wraps them in EOS/BOS tags
    # It also makes sure not to include the last bot tag "thinking..." since this is clearly not needed
    for count in range(len(conversation["user"])):
        enc += "<|BOS|>"
        enc += conversation["user"][count]
        enc += "<|EOS|>"
        if count != len(conversation["user"])-1:
            enc += "<|BOS|>"
            enc += conversation["bot"][count]
            enc += "<|EOS|>"
    reply = ts.test(enc)  # Sends encoder input to the test of the training skeleton and gets a response
    if reply[-10:] == "<|ENDGEN|>":
        reply = reply[:-10]
    await websocket.send(reply)


start_server = websockets.serve(get_msg_and_reply, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()