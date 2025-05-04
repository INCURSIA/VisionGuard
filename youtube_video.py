import yt_dlp

def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': 'video.mp4',  # Save the output as video.mp4
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=XF7YDi3OJ9U"
    download_video(url)
