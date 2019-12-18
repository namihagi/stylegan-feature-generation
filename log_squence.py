# confirm order functions

def out_log(mess):
    with open('log_sequence.txt', 'a') as fout:
        fout.write(mess + '\n')
