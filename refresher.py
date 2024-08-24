def mergeAlternately(word1, word2):
    word_len = len(word1)
    if len(word1) > len(word2):
        word_len = len(word2)

    out_word = ''

    for i in range(word_len):
        out_word += word1[i]
        out_word += word2[i]



    return out_word + word1[word_len:] + word2[word_len:]

print(mergeAlternately('abc', 'pqr'))