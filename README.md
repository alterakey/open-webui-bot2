# open-webui-bot2: Alternative bot for open-webui channels

open-webui-bot2 is an alternative bot implementation for use in open-webui channels. We are currently able to:

* Remember channel history
* Ignore "whisper" conversations (/whisper @target ...)
* Give custom system prompt
* Require mentions

## Building

```
docker build -t bot2 .
```

## Running

```
docker run --rm --link <container running open-webui>:api bot2
```
