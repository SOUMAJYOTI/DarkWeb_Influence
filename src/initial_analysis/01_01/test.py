input_list = ['all', 'this', 'happened', 'more', 'or', 'less']

# def find_ngrams(input_list, n):
print(list(zip(*[input_list[i:] for i in range(3)])))