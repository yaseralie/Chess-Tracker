# TikTok Live
Original Source: https://codesandbox.io/p/github/Trycatch-tv/TikTokLive/master

### Youtube Reference
#### TikTok Live Data
<p align="center">
  <a href="https://www.youtube.com/watch?v=BdRBXKvQPXE" target="_blank">
    <img src="https://img.youtube.com/vi/BdRBXKvQPXE/0.jpg" alt="YouTube Video Thumbnail" width="480" />
  </a>
</p>
Click the image above to watch the video

##### 1. Update & Install
```
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip curl
```


##### 2. Make Folder
```
mkdir tiktok-live-python
cd tiktok-live-python
```

##### 3. Prepare Python Virtual Env
```
python3 -m venv venv
source venv/bin/activate
pip install TikTokLive
```
Check version:
```
pip show TikTokLive
```

##### 4. Create example Python Code: app.py
```
from TikTokLive import TikTokLiveClient
from TikTokLive.events import ConnectEvent, CommentEvent, GiftEvent, LikeEvent

# === Input Unique ID TikTok secara dinamis ===
unique_id = input("Masukkan Unique ID (tanpa @): ").strip()

# Buat client dengan unique_id hasil input
client = TikTokLiveClient(unique_id=unique_id)

# Event connect
@client.on(ConnectEvent)
async def on_connect(event: ConnectEvent):
    print("? Connected to Room ID:", client.room_id)

# Event comment
@client.on(CommentEvent)
async def on_comment(event: CommentEvent):
    print(f">> {event.user.nickname}: {event.comment}")

# Event gift
@client.on(GiftEvent)
async def on_gift(event: GiftEvent):
    gift_name = getattr(event.gift, "name", "Unknown Gift")
    count = getattr(event.gift, "repeat_count", 1)
    print(f"?? {event.user.nickname} sent {gift_name} x{count}")

# Event like
@client.on(LikeEvent)
async def on_like(event: LikeEvent):
    total = getattr(event, "total_likes", None)
    if total is not None:
        print(f"?? {event.user.nickname} tapped like! Total likes: {total}")
    else:
        print(f"?? {event.user.nickname} tapped like! (+{event.count})")

if __name__ == '__main__':
    client.run()

```

##### 5. Run code
```
python app.py
```