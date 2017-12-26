FILE=$1

if [[ $FILE != "papers_video" ]]; then
    echo "Available datasets are: papers_video"
    exit 1
fi

URL=https://users.eecs.northwestern.edu/~mif365/deep_video_cs/datasets/$FILE.zip
ZIP_FILE=./download/$FILE.zip
TARGET_DIR=./download/$FILE/
wget -N $URL -O $ZIP_FILE --no-check-certificate
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./download/
rm $ZIP_FILE
