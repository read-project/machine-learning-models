# *******************************************************************************************
# Script usage: drive_wget.sh [destination folder]
# if no destination folder supplied, destination is .
#
# Complete pipeline:
# 1) tar folder with:	tar -czvf ReAD.tar.gz folder
#    where folder is the folder to tar
#
# 2) load .tar.gz file to Drive
#
# 3) go to Drive, share the file to all that has link and get the link
#    i.e https://drive.google.com/file/d/1LaV8ceOXkIqTLBpZflzOOIzCrtnUzEp2/view?usp=sharing
#        https://drive.google.com/file/d/153n76O459xlRneDPdZEbyhV7aaE2_agt/view?usp=share_link
# 4) Take file id and put in FILEID variable

#Get output folder as paramether
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    D_PATH="."
  else
    D_PATH=$1
fi

#Destination PATH
echo "Uncompress in $D_PATH"

FILEID=134wsX9oucR5djR2KI77Dsl5TQm0IQOUA #153n76O459xlRneDPdZEbyhV7aaE2_agt
FILENAME=$D_PATH/ReAD.tar.gz

# Get the file
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt

echo "Unzipping..."
# Unzip
tar -xf $FILENAME -C $D_PATH
echo "...done."

# Delete zip
rm $FILENAME


