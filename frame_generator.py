from asyncio.log import logger
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import argparse
from decouple import config
from pathlib import Path
from datetime import datetime
import logging
import time

my_parser = argparse.ArgumentParser(
    description='Frame Generator',
    epilog=' -- Project DragonFly -- '
)

my_parser.add_argument("IP", type=str)

args, unknown = my_parser.parse_known_args()


IP_ADDRESS = str(args.IP)

LOG_DIR = Path(config("LOG_DIR", default="logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = str(LOG_DIR / IP_ADDRESS) + ".log"

logging.basicConfig(filename=LOG_FILE,
                    filemode='a',
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG)

logging.info("INITIALISING")


CREDENTIALS = config("CREDENTIALS")
VIDEO_SRC = "rtsp://" + CREDENTIALS + "@" + IP_ADDRESS + ":554/media/video1"

FRAME_BASE_DIR = Path(config("FRAME_BASE_DIR", default="frame"))
FRAME_DIR = FRAME_BASE_DIR / IP_ADDRESS / datetime.now().strftime("%m%d")
FRAME_DIR.mkdir(parents=True, exist_ok=True)
FRAME_OUT = str(FRAME_DIR / "frame-")
RUN_HOUR = config("RUN_HOUR", cast=float, default=0.5)

DEFAULT_SCORECUT = 0.75

logging.info("Proccessing for " + IP_ADDRESS)

duration = config("FRAMES", cast=int, default=2000)


logging.info("Total accepeted frames required: " + str(duration))


logging.info("Initialising Blacklist")
BLACKLIST_DIR = Path("blacklist")
BLACKLIST_IMG = []
BLACKLIST_THRESHOLD = 0.75
if BLACKLIST_DIR.exists():
    for p in BLACKLIST_DIR.glob("**/*"):
        if p.is_file():
            BLACKLIST_IMG.append(cv2.imread(str(p), cv2.COLOR_BGR2GRAY))

logging.info("Backlist count:" + str(len(BLACKLIST_IMG)))


vidcap = cv2.VideoCapture(VIDEO_SRC)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success, image = vidcap.read()
grayB = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


i = 0
saved_frame_count = 0

percent = round(i/length, 3)

score_cut = DEFAULT_SCORECUT
STREAM_ERROR_MAX = 20
stream_error_count = 0


try:
    while(saved_frame_count < duration):
        i += 1
        score_cut += 0.001
        new_percent = round(i/length, 3)

        if new_percent != percent:
            percent = new_percent
            logging.info(percent)

        success, image = vidcap.read()
        if not success:
            logging.warning(f"Stream ended or error! {stream_error_count}")
            time.sleep(stream_error_count//2)
            stream_error_count += 1
            if stream_error_count < STREAM_ERROR_MAX:
                continue
            else:
                break

        stream_error_count = 0

        grayA = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        if score_cut <= score:
            continue

        blacklist_score = 0
        for img in BLACKLIST_IMG:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(grayA, imgGray, full=True)
            blacklist_score = (blacklist_score+score)/2

        if blacklist_score > BLACKLIST_THRESHOLD:
            logging.warning(
                "Frame blacklisted, skipping. Blacklist_Score=" + str(blacklist_score))
            continue

        logging.info(f"Saved frame {i} with %.4f similarity. BScore=%.2f count={saved_frame_count}" % (
            score*100, blacklist_score*100))

        outfile = FRAME_OUT+str(i)+".jpg"
        cv2.imwrite(outfile, image)
        saved_frame_count += 1
        score_cut = DEFAULT_SCORECUT
        grayB = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    logging.info("Successfully saved frames: " + str(saved_frame_count))
except Exception as e:
    logger.error(e)
