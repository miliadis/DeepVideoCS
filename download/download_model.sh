FILE=$1

if [[ $FILE != "1M_fc4_mask_trained" ]]; then
    echo "Available models are: 1M_fc4_mask_trained"
    exit 1
fi

URL=https://users.eecs.northwestern.edu/~mif365/deep_video_cs/models/$FILE/model_best.pth.tar
TAR_FILE=./download/model_best.pth.tar
TARGET_DIR=./download/models/$FILE/
wget -N $URL -O $TAR_FILE --no-check-certificate
mkdir -p $TARGET_DIR
mv $TAR_FILE $TARGET_DIR
