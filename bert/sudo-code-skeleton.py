# sudo code skeleton to map out the steps needed as I develop the process.


# load models, libs, etc / declare vars / etc

# import / request note data from zk

# prepare note data (extract note body text)

# model.encode(the_zk_text_data) 

# compute similarities

# map vector results to note titles (key / value pairs?)

# cluster similar documents based on user provided similarity strength variable?
# this would then be used to return groups of results, instead of one giant
# table
# see here: https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/clustering

# return results
