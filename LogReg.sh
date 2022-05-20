#!/bin/bash
# commandline arg: 
# bin for binary unigrams, wc for word count unigrams, big for bigrams
features=${1:-bin}
echo "${features} features (unigrams: bin-binary,wc-word count|bigrams: big)"
# rm VW/logreg.model
echo -e "\nTraining\n"
# cat VW/${features}_int_train.txt VW/${features}_int_wiki_top.txt | shuf > VW/train.vw
# cat int_train.txt | shuf > train.vw
# cat VW/${features}_int_train.txt | shuf > train.vw
# vw -i logreg_train_wtopx.model.4 --final_regressor logreg_train_wtopx2.model --loss_function logistic --oaa 2382 --holdout_off --probabilities --passes 75 -P 1.25 --kill_cache --cache_file cache --save_resume --save_per_pass -l 0.01 --adaptive < train.vw

# vw --final_regressor VW/logreg_train_wiki_top.model --loss_function logistic --oaa 2382 --holdout_off --probabilities --passes 75 -P 1.25 --kill_cache --cache_file cache --save_resume --save_per_pass -l 100 --adaptive < train.vw
# vw --final_regressor VW/logreg_wiki_top.model --loss_function logistic --oaa 2382 --holdout_off --probabilities --passes 75 -P 1.25 --kill_cache --cache_file cache --save_resume --save_per_pass -l 100 --adaptive < train.vw
# vw --final_regressor VW/logreg_wiki.model --loss_function logistic --oaa 2382 --holdout_off --probabilities --passes 75 -P 1.25 --kill_cache --cache_file cache --save_resume --save_per_pass -l 100 --adaptive < train.vw
# vw -f VW/logreg_test.model --loss_function logistic --oaa 2382 --holdout_off --probabilities --passes 7 -P 1.25 --kill_cache --cache_file cache --save_resume --save_per_pass -l 100 --adaptive < train.vw
# vw --final_regressor VW/logreg_train_wc.model --loss_function logistic --oaa 2382 --holdout_off --probabilities --passes 75 -P 1.25 --kill_cache --cache_file cache --save_resume --save_per_pass -l 100 --adaptive < train.vw
# vw --final_regressor VW/nn.model --nn 16 --oaa 2382 --holdout_off --passes 1 -P 1.25 --kill_cache --cache_file cache -l 0.1 < train.vw

echo -e "\nTesting\n"
shuf int_test_pos5.txt > test.vw
# shuf VW/${features}_int_dev.txt > VW/test.vw

awk '{print $1}' test.vw > y_true.txt
vw --testonly -i logreg_train_wtopx2.model.4 -P 1.25 --top 5 --predictions preds.txt < test.vw

# vw --testonly -i VW/logreg_wiki_top.model.75 -P 1.25 --predictions VW/preds.txt < VW/test.vw
# vw --testonly -i VW/logreg_train_wiki_top.model.50 -P 1.25 --predictions VW/preds.txt < VW/test.vw
# vw --testonly -i VW/logreg_wiki.model -P 1.25 --predictions VW/preds.txt < VW/test.vw
# vw --testonly -i VW/logreg_train_wc.model -P 1.25 --predictions VW/preds.txt < VW/test.vw
# vw --testonly -i VW/logreg_test.model.10 --predictions VW/preds.txt < VW/test.vw
# vw --testonly -i VW/nn.model --predictions VW/preds.txt < VW/test.vw

awk '{print $1}' preds.txt > y_pred.txt
awk {'printf ("%s %s\n", $1, $2)'} test.vw > labels.txt
sed -i 's/['\'\|]'//g' labels.txt
pr -m -t labels.txt y_pred.txt > results.txt
total=$(cat y_pred.txt | wc -l)
# echo "$(pr -m -t y_pred.txt y_true.txt)"
wrong=$(awk 'NR == FNR { lines[NR] = $x } NR != FNR && lines[FNR] != $x {print}' y_true.txt y_pred.txt | wc -l)
correct=$(($total-$wrong))         
accuracy=$(echo "print(${correct}/${total})" | python3)
echo -e "\nThe accuracy is ${accuracy} (${correct}/${total} correct)\n"

