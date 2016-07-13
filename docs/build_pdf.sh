#/usr/bin/env bash

rm -rf _build
sphinx-build -b latex . _build

if [ $? -eq 0 ];
    then
    if [ ! -z $1 ] && [ $1 == "show" ]
        then
            cd _build
            make
            cd ..
            if [ $? -eq 0 ];
                then
                    evince _build/cbcpost.pdf &
            fi
    fi
fi
