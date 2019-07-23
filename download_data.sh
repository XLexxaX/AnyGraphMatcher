#!/usr/bin/env bash

function download_and_extract {
    var=$1
    filename="${var##*/}"
    if [ ! -f "$2/${filename}.extracted" ]; then
        mkdir -p "$2"
        wget -O $2/${filename} $1 && \
            tar xvzf $2/${filename} -C $2 && \
            rm "$2/${filename}" && \
            touch "$2/${filename}.extracted" # can uncomment the rm command to not remove the compressed file
    else
        echo "File '$2/${filename}.extracted' already there; not retrieving."
    fi
}

function copy_and_extract {
    var=$1
    filename="${var##*/}"
    if [ ! -f "$2/${filename}.extracted" ]; then
        mkdir -p "$2"
        cp $1 $2/${filename}&& \
            tar xvzf $2/${filename} -C $2 && \
            rm "$2/${filename}" && \
            touch "$2/${filename}.extracted" # can uncomment the rm command to not remove the compressed file
    else
        echo "File '$2/${filename}.extracted' already there; not retrieving."
    fi
}

function apply_wmt_process_scripts {
    if [ ! -f "$1.tokenized.lowercased.processed.colloq.unique" ]; then
        data/wmt11/scripts/tokenizer.perl -l $2 <"$1"> "$1.tokenized"
        cat "$1.tokenized" | data/wmt11/scripts/lowercase.perl > "$1.tokenized.lowercased"
        rm "$1.tokenized"
        python3 cle/eval/processwmt.py preprocess "$1.tokenized.lowercased" > "$1.tokenized.lowercased.processed"
        python3 cle/eval/processwmt.py collocation "$1.tokenized.lowercased.processed" > "$1.tokenized.lowercased.processed.colloq"
        rm "$1.tokenized.lowercased.processed"
        sort -u "$1.tokenized.lowercased.processed.colloq" > "$1.tokenized.lowercased.processed.colloq.unique"
    else
        echo "File '$1.tokenized.lowercased.unique' already there; not applying any wmt process scripts."
    fi
}

function downloadseals {
    if [ ! -f data/oaei/$1.extracted ]; then
        python3 cle/eval/downloadseals.py http://repositories.seals-project.eu/tdrs/ $1 $2 data/oaei/ && touch data/oaei/$1.extracted
    else
        echo "File 'data/oaei/$1.extracted' already there; not retrieving."
    fi
}


#download dbpedia related stuff
for LANGUAGE in en fr ;do #ja zh; do # en fr ja zh
    for TYPE in interlanguage_links infobox_properties; do #interlanguage_links infobox_properties; do
        wget -nc --directory-prefix=data/dbpedia/${LANGUAGE} http://downloads.dbpedia.org/2016-10/core-i18n/${LANGUAGE}/${TYPE}_${LANGUAGE}.ttl.bz2
    done
    #download mapping 2016-10 commit: 677cd4ea571830b3fb9a226a9ae0bf7cdab8acfb
    wget -nc --directory-prefix=data/dbpedia/${LANGUAGE} \
        https://raw.githubusercontent.com/dbpedia/extraction-framework/677cd4ea571830b3fb9a226a9ae0bf7cdab8acfb/mappings/Mapping_${LANGUAGE}.xml
done


#download jape data
download_and_extract https://github.com/nju-websoft/JAPE/raw/89bbb661dac8908ee4258863051ff569a4483b20/data/dbp15k.tar.gz data/jape
#or use http://ws.nju.edu.cn/jape/data/DBP15k.tar.gz   and   http://ws.nju.edu.cn/jape/data/DBP100k.tar.gz


#dbkwik
download_and_extract http://data.dws.informatik.uni-mannheim.de/dbkwik/KGs_for_gold_standard.tar.gz data/dbkwik/KGs_for_gold_standard
for filename in darkscape~oldschoolrunescape heykidscomics~dc marvel~dc \
        marvel~heykidscomics memory-alpha~memory-beta memory-alpha~stexpanded \
        memory-beta~stexpanded runescape~darkscape runescape~oldschoolrunescape ; do
    wget -nc --directory-prefix=data/dbkwik/gold https://raw.githubusercontent.com/sven-h/dbkwik/18bed5b2e2e338a9e18b90ea07270def5f7a6c29/e_gold_mapping_interwiki/gold/${filename}~evaluation.xml
done
#wget -nc --directory-prefix=data/dbkwik/gold https://raw.githubusercontent.com/sven-h/dbkwik/master/e_gold_mapping_interwiki/gold/darkscape~oldschoolrunescape~evaluation.xml


#download nlp texts (wmt11) - http://www.statmt.org/wmt11/
download_and_extract http://www.statmt.org/wmt11/training-monolingual-news-2011.tgz data/wmt11 # tar file contains folder training-monolingual
download_and_extract http://www.statmt.org/wmt08/scripts.tgz data/wmt11 # tar file contains folder scripts
apply_wmt_process_scripts data/wmt11/training-monolingual/news.2011.en.shuffled en
apply_wmt_process_scripts data/wmt11/training-monolingual/news.2011.fr.shuffled fr
apply_wmt_process_scripts data/wmt11/training-monolingual/news.2011.de.shuffled de

#copy nlp gold standard:
copy_and_extract lexicon_wmt11.tar.gz data/wmt11
copy_and_extract amazon_data.tar.gz data/amazon_data

#download eval word sim
for filename in EN-MC-30.txt EN-MEN-TR-3k.txt EN-MTurk-287.txt EN-MTurk-771.txt \
        EN-RG-65.txt EN-RW-STANFORD.txt EN-SIMLEX-999.txt EN-SimVerb-3500.txt EN-VERB-143.txt \
        EN-WS-353-ALL.txt EN-WS-353-REL.txt EN-WS-353-SIM.txt EN-YP-130.txt;do
    wget -nc --directory-prefix=data/wordsim https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/56ebe3699df6e745d506fb3cd97d4f7b1e615d32/data/word-sim/${filename}
done


#download seals datasets
downloadseals conference conference-v1
downloadseals anatomy_track anatomy_track-default
downloadseals de-en de-en-v2
