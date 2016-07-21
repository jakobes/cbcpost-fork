#/usr/bin/env bash
rm -rf rst_programmers_reference
rm -rf _build
sphinx-build -b latex . _build
python create_booktabs.py

if [ $? -eq 0 ];
    then
    if [ ! -z $1 ] && [ $1 == "show" ]
        then
            cd _build
            make
            if [ $? -eq 0 ];
                then
                    evince cbcpost.pdf &
            fi
            cd ..
    fi
fi
