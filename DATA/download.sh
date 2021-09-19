#!/usr/bin/env bash
# https://www.quora.com/How-do-I-download-a-very-large-file-from-Google-Drive/answer/Shane-F-Carr
# 1. Go to OAuth 2.0 Playground (https://developers.google.com/oauthplayground/)
# 2. In the “Select the Scope” box, scroll down, expand “Drive API v3”, and select https://www.googleapis.com/auth/drive.readonly
# 3. Click “Authorize APIs” and then “Exchange authorization code for tokens”. Copy the “Access token”; you will be needing it below.
Auth=""
URLS=(
    "1UBPNwFEVhIZCOEpu4hTqPji57XRg85UO"
    # "14d2ttsuEUFXsxx-KRWJMsFhQGrYOJcpH"
    # "1acIBqcPVX5QXXXV9u8_yDPtCgfsdEJDV"
    # "1qbK88SAKxqjMUybkMeIjrJWnNAZyE8V0"
    # "11fNraDQs-LiODDxyV5ZW0Slf3XuDq5Cf"
    # "1C5qK1FckA702nsYcXwmGdzlMmHg1F_ot"
    # "1nbdYR5VqcTbLpOB-9cICKCgsLAs7fVzd"
    # "1Z3hSiP7fsR3kf8fjQYzIa07jmw4KXNnw"
    # "1iwOO60H_1DdcU1zSIqBv_y5PwXbJHjQ2"
)
FILES=(
    "MUSTC_v2.0_en-de.tar.gz"
    # "MUSTC_v1.0_en-es.tar.gz"
    # "MUSTC_v1.0_en-fr.tar.gz"
    # "MUSTC_v1.0_en-it.tar.gz"
    # "MUSTC_v1.0_en-nl.tar.gz"
    # "MUSTC_v1.0_en-pt.tar.gz"
    # "MUSTC_v1.0_en-ro.tar.gz"
    # "MUSTC_v1.0_en-ru.tar.gz"
    # "MUSTC_v1.2_en-zh.tar.gz"
)

if [ -z "$Auth" ]; then
    echo "Set Auth token first: export Auth=\"...\""
    exit 1
fi
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        curl -H "Authorization: Bearer $Auth" "https://www.googleapis.com/drive/v3/files/${url}?alt=media" -o $file
        if [ -f $file ]; then
            echo "$file successfully downloaded."
        else
            echo "$file not successfully downloaded."
            exit -1
        fi
    fi
done