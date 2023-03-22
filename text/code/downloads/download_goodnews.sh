#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

FILENAME=

download_goodnews() {
    curl "https://cvcuab-my.sharepoint.com/personal/abiten_cvc_uab_cat/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fabiten%5Fcvc%5Fuab%5Fcat%2FDocuments%2Fgoodnews%2F$1" \
        -H 'authority: cvcuab-my.sharepoint.com' \
        -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
        -H 'accept-language: ko-KR,ko;q=0.9,en-GB;q=0.8,en;q=0.7,en-US;q=0.6' \
        -H 'cookie: FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEyLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uIzYyM2RjODdmNWZhOWVkYTBjM2NjNjE5NTNhYWFjODdjYjk5MzQ1NWQxNWVhODZmMTg4MjhmYzc5YTU4NGMzZjgsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jNjIzZGM4N2Y1ZmE5ZWRhMGMzY2M2MTk1M2FhYWM4N2NiOTkzNDU1ZDE1ZWE4NmYxODgyOGZjNzlhNTg0YzNmOCwxMzI5NjQ1NDY3NDAwMDAwMDAsMCwxMzI5NjU0MDc3NDA3ODEyODQsMC4wLjAuMCwyNTgsMWE1ZWIxYjQtYTU2Ni00MmMyLTg3ZmMtMTdhMzY5YjQwOTE4LCwsNzAzZTNiYTAtNjAxNi00MDAwLTNiOTctZDQ2ODAwNDc3ZTM1LDcwM2UzYmEwLTYwMTYtNDAwMC0zYjk3LWQ0NjgwMDQ3N2UzNSxVNUppeWNyU3lFcU1Zc21RSURDMDlRLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCxybW9qdWZpcmZWTzlVUmMrQUhQMVpBZWpabXdHa2V1emZEQnpFd1ZLK0tFVnEvMDNnRDdsNS9vS0c5QmVPVDJYckpjNk5XM2JYT3hTbHFqS0tqVWVvZXZNTzVIb2pMc20yUUUySldGRy8zY1VRaGtQVnN3U0YxcytsTWhhbDhTNnhFT0E5aUMyWUlYcjNMSUp4SXZ4c1dyN1JTT0gzVkNYaUM5WlNuM0ZHcFpxVkc5RkNzbmpmTWw2QVNyeHRXaE42YmhxMzJwSkEwUGQyTlE4Unh4SnBtb3hieDZtM1ZpeE5Wa2FUclBRV09ZT0liT3lPUWtKUUpOaHRYOU8vNk5HSjY2VnBxVWM1Wk4vK1ZZM2VuTER3ZGt4QUtqVWZ1ZitqMzJwV1c4NUVBOFdnU1cwOTBGSmlCaXhkRDZtc29LMU9NMk8yRHUvREVGWnhZQVIxS0t6TEE9PTwvU1A+; KillSwitchOverrides_enableKillSwitches=; KillSwitchOverrides_disableKillSwitches=' \
        -H "referer: https://cvcuab-my.sharepoint.com/personal/abiten_cvc_uab_cat/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fabiten%5Fcvc%5Fuab%5Fcat%2FDocuments%2F$1&parent=%2Fpersonal%2Fabiten%5Fcvc%5Fuab%5Fcat%2FDocuments%2Fgoodnews&ga=1" \
        -H 'sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="101", "Google Chrome";v="101"' \
        -H 'sec-ch-ua-mobile: ?0' \
        -H 'sec-ch-ua-platform: "Windows"' \
        -H 'sec-fetch-dest: iframe' \
        -H 'sec-fetch-mode: navigate' \
        -H 'sec-fetch-site: same-origin' \
        -H 'service-worker-navigation-preload: true' \
        -H 'upgrade-insecure-requests: 1' \
        -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36' \
        --compressed --output $SCRIPT_DIR/$2
}


curl 'https://cvcuab-my.sharepoint.com/personal/abiten_cvc_uab_cat/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fabiten%5Fcvc%5Fuab%5Fcat%2FDocuments%2Fgoodnews%2Fnews%5Fdataset%2Ejson' \
    -H 'authority: cvcuab-my.sharepoint.com' \
    -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
    -H 'accept-language: ko-KR,ko;q=0.9,en-GB;q=0.8,en;q=0.7,en-US;q=0.6' \
    -H 'cookie: FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEyLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uI2QyYzhlNmM3MWZmYTljODc0ZDNiZGVlMDA3ZGQxNDY4OTc0ZWFkZWM2OWRhMTBiYWExMjllYzg5ZDUxMGJlMzksMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jZDJjOGU2YzcxZmZhOWM4NzRkM2JkZWUwMDdkZDE0Njg5NzRlYWRlYzY5ZGExMGJhYTEyOWVjODlkNTEwYmUzOSwxMzI5NjUzMDk3MjAwMDAwMDAsMCwxMzI5NjYxNzA3Mjc4MDA0MjMsMC4wLjAuMCwyNTgsMWE1ZWIxYjQtYTU2Ni00MmMyLTg3ZmMtMTdhMzY5YjQwOTE4LCwsMzM4NzNiYTAtMzBiNS00MDAwLTNiOTctZDYxMjhkYmUxYzU2LDMzODczYmEwLTMwYjUtNDAwMC0zYjk3LWQ2MTI4ZGJlMWM1NiwzVUJ0alBSRURrU3NYOFpvMFRBeEJ3LDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCwzdk5yN2huWjYyREdsZEFrUm9SV2RYY3U2ZjVERzlEVGVZMmtvVnFYa2R4NEZ3elJ4UGtpV280dUtGZk1uQ2ZPSllWSGZzVmc0Y2NzWk5DVFplZVA2d0puUW9wQTFtNFhGQlR5VVZZZEo0NHltMC9ZQU9UemNCamcvWkRVakFwYlBPcVZ6N1ppandwaVBDMUVmWWsxbmVGTXE2ZWNaS25ISndDbDV3a2h4WVIxR1B3Wi9VU1cwbE1sOTNIb2M1RnVqRHZHMTlQblgwSUhFYlYvRWM5elcwZWNvekh2VjlRRVNOSEF4TDArZFdhMVhsU04vTVBpMGNpU1E2Z0hXRzV6ZEY0dE5CUUdyQ3d4T1ZNdytHeitzaExsQVRIbEJaT0hoL1plRExKMmhTTjNYcFJMclRiNFVkeTVoMGp1SVlHL3VWYXMyT0NRVm9KMVpGMzA5eWtFT1E9PTwvU1A+; KillSwitchOverrides_enableKillSwitches=; KillSwitchOverrides_disableKillSwitches=' \
    -H 'referer: https://cvcuab-my.sharepoint.com/personal/abiten_cvc_uab_cat/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fabiten%5Fcvc%5Fuab%5Fcat%2FDocuments%2Fgoodnews%2Fnews%5Fdataset%2Ejson&parent=%2Fpersonal%2Fabiten%5Fcvc%5Fuab%5Fcat%2FDocuments%2Fgoodnews&ga=1' \
    -H 'sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'sec-ch-ua-platform: "macOS"' \
    -H 'sec-fetch-dest: iframe' \
    -H 'sec-fetch-mode: navigate' \
    -H 'sec-fetch-site: same-origin' \
    -H 'service-worker-navigation-preload: true' \
    -H 'upgrade-insecure-requests: 1' \
    -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36' \
    --compressed --output $SCRIPT_DIR/news_dataset.json

# download_goodnews "captioning%5Fdataset%2Ejson" "goodnews.json"
# download_goodnews "img%5Fsplits%2Ejson" "splits.json"
# download_goodnews 'news%5Fdataset%2Ejson' "news_dataset.json"
