def radixSort(words, alph, max_word_len):
    def countingSort(words, pos):
    
        # convert between letter and index
        ltoi = {l:i + 1 for i,l in enumerate(alph)}
        itol = {i + 1:l for i,l in enumerate(alph)}

        # add empty char
        ltoi[''] = 0
        itol[0] = ''

        # Track count of each letter
        count = [0] * (len(alph) + 1)
        for s in words:
            if pos < len(s):
                l = s[pos]
            else:
                l = ''

            count[ltoi[l]] += 1

        # Update counts to become positions
        for i in range(1, len(count)):
            count[i] += count[i-1]

        # construct output
        out = [None] * len(words)

        for i in range(len(words)-1, -1, -1):
            if pos < len(words[i]):
                l = words[i][pos]
            else:
                l = ''

            li = ltoi[l]
            count[li] -= 1
            curloc = count[li]
            out[curloc] = words[i]

        return out
    
    for i in range(max_word_len, -1, -1):
        words = countingSort(words, i)

    return words