from __future__ import annotations
from typing import TYPE_CHECKING

import asyncio
from collections import deque
import aiohttp
import socketio

if TYPE_CHECKING:
  from typing import NoReturn, AsyncIterator, Mapping, Any, Deque, Optional, List

  Event = Mapping[str, Any]
  ChannelEvent = Event
  Message = Mapping[str, Any]


class UI:
  is_debugging = False

  def stderr(self, msg: str) -> None:
    from sys import stderr
    print(msg, file=stderr)

  def fatal(self, msg: str) -> NoReturn:
    from sys import exit
    self.stderr(f'fatal: {msg}')
    exit(2)

  def info(self, msg: str) -> None:
    self.stderr(msg)

  def warn(self, msg: str) -> None:
    self.info(msg)

  def debug(self, msg: str) -> None:
    if self.is_debugging:
      self.stderr(msg)


ui = UI()


class ChannelLog:
  def __init__(self, filter_: ChannelLogFilter, prompt: str) -> None:
    self._content: Deque[ChannelEvent] = deque()
    self._prompt = prompt
    self._filt = filter_

  def __bool__(self) -> bool:
    return bool(self._content)

  async def fetch(self, url: str, token: str, channel_id: str) -> None:
    async with aiohttp.ClientSession() as sess:
      async with sess.get(
        f"{url}/api/v1/channels/{channel_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
      ) as response:
        if response.status == 200:
          msgs = await response.json()
          self._content.clear()
          for msg in msgs:
            del msg['user']['profile_image_url']
            self._content.append(msg)
        else:
          # Optional: Handle errors or return raw response text
          ui.warn(f'fetch failed: {response.status}; {await response.text()}')

  async def build_graph(self, self_id: str, to_msg_ev: ChannelEvent) -> AsyncIterator[Any]:
    yield {"role": "system", "content": f"{self._prompt}"}

    for ev in sorted(self._content, key=lambda x: x['created_at']):
      if ev['id'] == to_msg_ev['message_id']:
        break
      if self._filt.should_be_unheared(ev['content']):
        continue
      if ev['user']['id'] == self_id:
        yield {"role": "assistant", "content": ev['content']}
      else:
        yield {'role': 'user', 'content': ev['content']}

    yield {"role": "user", "content": to_msg_ev['data']['data']['content']}

class ChannelLogFilter:
  _name: Optional[str] = None

  def __init__(self, require_mention: bool) -> None:
    self._require_mention = require_mention

  def reset(self) -> None:
    self._name = None

  def set_name(self, name: str) -> None:
    self._name = name

  def should_be_unheared(self, content: str) -> bool:
    if self._name is not None:
      if f'@{self._name}' not in content:
        if self._require_mention or content.startswith('/whisper'):
          return True
    return False

class Bot2:
  def __init__(self, url: str, model_id: str, token: str, prompt: Optional[str], require_mention: bool) -> None:
    self._url = url
    self._model_id = model_id
    self._token = token
    self._filt = ChannelLogFilter(require_mention)
    self._log = ChannelLog(self._filt, prompt if prompt is not None else '')
    self._user_id = None
    self._sio = socketio.AsyncClient(logger=False, engineio_logger=False)
    self._sio.on("connect", self._on_connect)
    self._sio.on("disconnect", self._on_disconnect)
    self._sio.on("channel-events", self._on_channel_events)

  async def start(self) -> None:
    ui.info(f"Connecting to {self._url}...")
    await self._sio.connect(
      self._url, socketio_path="/ws/socket.io", transports=["websocket"]
    )
    ui.info("Connection established!")
    await self._sio.emit("user-join", {"auth": {"token": self._token}}, callback=self._on_user_join)
    await self._sio.wait()

  async def _on_user_join(self, data: ChannelEvent) -> None:
    self._user_id = data['id']
    self._filt.set_name(data['name'])

  async def _on_connect(self) -> None:
    ui.info("Connected!")

  async def _on_disconnect(self) -> None:
    self._user_id = None
    self._filt.reset()
    ui.info("Disconnected from the server!")

  async def _openai_chat_completion(self, messages: List[Message]) -> Any:
    ui.debug(f'completing on {messages}')
    payload = {
      "model": self._model_id,
      "messages": messages,
      "stream": False,
    }

    async with aiohttp.ClientSession() as session:
      async with session.post(
        f"{self._url}/api/chat/completions",
        headers={"Authorization": f"Bearer {self._token}"},
        json=payload,
      ) as response:
        if response.status == 200:
          return await response.json()
        else:
          # Optional: Handle errors or return raw response text
          return {"error": await response.text(), "status": response.status}

  async def _on_channel_events(self, data: ChannelEvent) -> None:
    if self._user_id is None:
      return

    ch = data['channel_id']
    await self._log.fetch(self._url, self._token, ch)

    if data["user"]["id"] == self._user_id:
      # Ignore events from the bot itself
      return

    if data["data"]["type"] == "message":
      if self._filt.should_be_unheared(data['data']['data']['content']):
        return

      ui.debug(f'{data["user"]["name"]}: {data["data"]["data"]["content"]}')

      await self._send_typing(ch)

      # OpenAI API coroutine
      # This uses naive implementation of OpenAI API, that does not utilize the context of the conversation
      openai_task = self._openai_chat_completion(
        [x async for x in self._log.build_graph(self._user_id, data)]
      )

      try:
        response = await self._send_typing_until_complete(
          ch, openai_task
        )

        if response.get("choices"):
          completion = response["choices"][0]["message"]["content"]
          await self._send_message(ch, completion)
        else:
          ui.debug(response)
          await self._send_message(
            ch, "I'm sorry, I don't understand."
          )
      except Exception as e:
        await self._send_message(
          ch,
          f"Something went wrong while processing your request. [{e}]",
        )

  async def _send_typing_until_complete(self, channel_id: str, coro: Any) -> Any:
    task = asyncio.create_task(coro)  # Begin the provided coroutine task
    try:
      while not task.done():
        await self._send_typing(channel_id)
        await asyncio.sleep(1)
      return await task
    except Exception:
      task.cancel()
      raise

  async def _send_message(self, channel_id: str, message: str) -> Any:
    url = f"{self._url}/api/v1/channels/{channel_id}/messages/post"
    headers = {"Authorization": f"Bearer {self._token}"}
    data = {"content": str(message)}

    async with aiohttp.ClientSession() as session:
      async with session.post(url, headers=headers, json=data) as response:
        if response.status != 200:
          # Raise an exception if the request fails
          raise aiohttp.ClientResponseError(
            request_info=response.request_info,
            history=response.history,
            status=response.status,
            message=await response.text(),
            headers=response.headers,
          )
        # Return response JSON if successful
        return await response.json()

  async def _send_typing(self, channel_id: str) -> None:
    await self._sio.emit(
      "channel-events",
      {
        "channel_id": channel_id,
        "data": {"type": "typing", "data": {"typing": True}},
      },
    )

  @classmethod
  def should_be_unheared(cls, name: str, content: str) -> bool:
    if content.startswith('/whisper') and f'@{name}' not in content:
      return True
    return False


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('endpoint', default='http://api:8080', nargs='?')
  parser.add_argument('--model', '-m', required=True)
  parser.add_argument('--token', '-t', required=True)
  parser.add_argument('--prompt', '-p')
  parser.add_argument('--require-mention', '-r', default=False, action='store_true')
  parser.add_argument('--debug', '-d', action='store_true')
  args = parser.parse_args()

  if args.debug:
    ui.is_debugging = True

  try:
    asyncio.run(Bot2(url=args.endpoint, model_id=args.model, token=args.token, prompt=args.prompt, require_mention=args.require_mention).start())
  except asyncio.CancelledError:
    pass
