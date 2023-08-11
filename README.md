# Reelsfy - Reels Clips Automator

Introducing Isabella Reels, the intelligent core behind Reelsfy. Inspired by the popularity of Instagram Reels, Isabella is here to transform the way you create content. She's the sister bot to Marcelo Resenha, known for assisting with YouTube video editing. With Isabella's capabilities, you can now effortlessly turn longer videos into engaging Instagram Reels.

Reelsfy is an advanced, AI-powered tool that automates the process of creating Instagram Reels from longer videos. Isabella uses a combination of computer vision to track faces, the GPT model to identify the most engaging parts of the video, and the Whisper ASR system to generate subtitles. This open-source project is perfect for content creators looking to streamline their workflow and optimize content for virality.

## Features

- Converts horizontal videos into vertical Reels, perfect for Instagram
- Downloads videos directly from YouTube or uses local video files
- Uses GPT models to identify and cut the most viral sections of the video
- Employs computer vision to track faces during the video editing process
- Generates captions using the Whisper ASR system
- Uses GPU for faster processing (optional)

## Prerequisites

- Anaconda >= 22.11.1
- Python >= 3.11
- FFMPEG >= 4.4.2
- OpenAI API Key
- A GPU is optional but recommended for faster processing
- Developed on Ubuntu 22.04


## Installation

1. Clone the git repository:

```
$ git clone https://github.com/eddieoz/reels-clips-automator.git
```

2. Create and activate a new conda environment:

```
$ conda create -n reels-clips-automator
$ conda activate reels-clips-automator
```

3. Navigate to the cloned repository's folder:

```
$ cd folder
```

4. Install the required dependencies:

```
$ python -m pip install -r requirements.txt
$ python -m pip install utils/auto-subtitle
```

5. Create a `.env` file in the root directory of the project and include your OpenAI API Key:

```
OPENAI_API_KEY='Your-OpenAI-API-key-here'
```

## Usage

To see the help:

```
$ python reelsfy.py --help
```

For a video from YouTube:

```
$ python reelsfy.py -v <youtube video url>
```

For a local file:

```
$ python reelsfy.py -f <video file>
```

Please note that videos should be approximately 20 minutes long due to the total token limit of the gpt-3.5-turbo-16k model.

## Support

For any queries or support, feel free to reach out:

- Twitter: @eddieoz
- YouTube: @eddieoz
- Zaps to Nostr: eddieoz@sats4.life
- Sats to eddieoz@zbd.gg

## Contributions

Contributions to the project are welcome! Feel free to check out the code and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project was inspired by the work of [NisaarAgharia's AI-Shorts-Creator](https://github.com/NisaarAgharia/AI-Shorts-Creator).
