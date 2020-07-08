
import json
import numpy as np

HTTPS_token = '14120' 
RT_token = '56898'
at_token = '137'
hashtag_token = '108'
gt_token = '135'
lt_token = '133'
amp_token = '111'
question_token = '136'
esclamation_token = '106'
period_token = '119'
two_periods_token = '131'
coma_token = '117'
dollar_token = '109'
period_coma_token = '132'
parenthesis_open_token = '113'
parenthesis_closed_token = '114'
star_token = '115'
slash_token = '120'
line_token = '118'
underscore_token = '168'
tilde_token = '198'
virgolette_token = '107'
square_parenthesis_open_token = '164'
square_parenthesis_closed_token = '166'
unk_token = '100'
others_tokens = ['11733', '12022']

special_tokens_list = [
    at_token,
    hashtag_token,
    gt_token,
    lt_token,
    amp_token,
    question_token,
    esclamation_token,
    period_token,
    coma_token,
    dollar_token,
    period_coma_token,
    two_periods_token,
    parenthesis_open_token,
    parenthesis_closed_token,
    star_token,
    slash_token,
    line_token,
    underscore_token,
    tilde_token,
    virgolette_token,
    square_parenthesis_open_token,
    square_parenthesis_closed_token,
    unk_token
]

special_tokens = {
    'https': HTTPS_token,
    'RT': RT_token,
    '@': at_token,
    '>': gt_token,
    '<': lt_token,
    '&': amp_token,
    '?': question_token,
    '!': esclamation_token,
    '.': period_token,
    ':': two_periods_token,
    '#': hashtag_token,
    ',': coma_token,
    '$': dollar_token,
    ';': period_coma_token,
    '(': parenthesis_open_token,
    ')': parenthesis_closed_token,
    '*': star_token,
    '/': slash_token,
    '-': line_token,
    '_': underscore_token,
    '~': tilde_token,
    '"': virgolette_token,
    '[': square_parenthesis_open_token,
    ']': square_parenthesis_closed_token,
    '[UNK]': unk_token
}


# convert list of strings to list of int
f_to_int = lambda x: int(x)
f_int = lambda x: list(map(f_to_int, x.split('\t')))

    
# save tweet_id along with its corresponding tokens
def save_tweet(identifier, text_tokens, output_file):
    string = identifier + ',' + '\t'.join(map(str, text_tokens)) + '\n'
    output_file.write(string)
    

# save mentions, hashtags or links
def save(identifier, text_tokens, text, mapped, count, output_file):
    for i in range(len(text_tokens)):
        text_tokens[i] = '\t'.join(map(str, text_tokens[i]))
    
    # each mentions is separated by a ";"
    # each token in a mention is separated by a "\t"
    string = identifier + '\x01' + str(count) + '\x01' + ';'.join(text_tokens) + '\x01' + ''.join(text) + '\x01' + '\t'.join(map(str, mapped)) + '\n'
    
    output_file.write(string)
    
    
def save_tweet_length(tweet_id, length, output_file):
    output_file.write(tweet_id + ',' + str(length) + '\n')
    
    

def split_line(l):
    l = l.split(',')
    t_id = l[0]
    t_list = l[1].split('\t')  # replace("\\n",'').replace("\\t",'\t')
    
    return t_id, t_list


def load_mapping(path):
    with open(path, 'r') as json_file:
        mapping_dict = json.loads(json_file.read())
    key, current_mapping =  max(mapping_dict.items(), key = lambda x: x[1])
    
    print("Loaded mapping : ", path)
    
    return mapping_dict, current_mapping, key


def save_mapping(path, mapping_dict):
    json_mapping = json.dumps(mapping_dict)
    with open(path, 'w+') as f:
        f.write(json_mapping)
    
    print("Saved mapping : ", path)


def map_to_unique_ids(_list, _dict, _current_mapping):
    mapped = []
    for elem in _list:
        if elem not in _dict:
            _dict[elem] = _current_mapping
            _current_mapping += 1
        
        mapped.append(_dict[elem])
    
    return mapped, _current_mapping


def read_sentences(input_file, lines_num, header_first_line):
    
    tweet_ids = []
    sentences = []
    
    row = 0
    
    with open(input_file, "r", encoding='utf-8') as reader:
        
        if header_first_line:
            reader.readline()  # ignore the first line since it contains the CSV header
        
        while True:
            
            #if row % 100000 == 0:
            #    print("\tReading line: ", row)
                
            if row == lines_num:
                print("\tLines : ", row)
                return tweet_ids, sentences
            
            line = reader.readline()

            if not line:
                break
            
            # produce embeddings not for all the rows
            #if row % 100 == 0:

            line = line.strip().split(',')
            tweet_id = line[0]
            input_ids = f_int(line[1])

            tweet_ids.append(tweet_id)
            sentences.append(input_ids)

            row += 1
            
    print("\tLines : ", row)
    
    return tweet_ids, sentences


# to reconstruct hashtags and mentions text
# return a single string without spaces
# @param _list : list containing lists of tokens (one list per each hashtag/mention)
def decode_hashtags_mentions(tokenizer, _list):
    strings_list = []

    for elem in _list:
        elem = tokenizer.decode(elem)
        elem = elem.replace(' ', '')

        strings_list.append(elem)  # otherwise last string not added

    return strings_list


def replace_escaped_chars(line):
    gt_string = "111\t175\t10123\t132" # "& gt ;"
    lt_string = "111\t43900\t132" # "& lt ;"
    amp_string = "111\t10392\t10410\t132" # "& amp ;"

    if gt_string in line:
        line = line.replace(gt_string, gt_token)
    if lt_string in line:
        line = line.replace(lt_string, lt_token)
    if amp_string in line:
        line = line.replace(amp_string, amp_token)
        
    return line


# return text_tokens, mentions_list, mentions_count
# in case the tweet is a retweet
def get_RT_mentions(tokens, mentions):

    length = len(tokens)-1
    
    i = 2  # exclude CLS and the 56898 ('RT') token
    while tokens[i] != special_tokens[':'] and i < length:
        i += 1

    #print('i: ' + str(i))

    mentions.append(tokens[2:i])
    #mentions.append('102\n') # append SEP \n

    tokens = tokens[i+1:]
    tokens.insert(0, '101')   # insert CLS at beginning
    
    return tokens, mentions


def get_remove_mentions_hashtags(tokenizer, tokens, mentions, hashtags):
    
    found_initial = False
    
    mask = []
    
    initial_index = 0
    final_index = 0
    is_mention = False
    
    for i in range(len(tokens)):
        
        t = tokens[i]
        
        if found_initial and i == initial_index+1:
            mask.append(False)
        
        elif found_initial and i > initial_index+1:
            decoded_t = tokenizer.convert_tokens_to_strings([t])[0]
            if '##' in decoded_t:
                mask.append(False)
            elif '_' == decoded_t:
                mask.append(False)
            elif tokenizer.convert_tokens_to_strings([tokens[i-1]])[0] == '_':
                mask.append(False)
            else:
                final_index = i
                if is_mention:
                    mentions.append(tokens[initial_index:final_index])
                else:
                    hashtags.append(tokens[initial_index:final_index])

                found_initial = False
                # mask.append(True)
    
                
        if not found_initial and (t == special_tokens['@'] or t == special_tokens['#']):
            if t == special_tokens['@']:
                is_mention = True
            elif t == special_tokens['#']:
                is_mention = False
                
            initial_index = i
            found_initial = True
            mask.append(False)
            
        elif not found_initial:
            mask.append(True)
            
            #print(decoded_t)
    tokens_arr = np.array(tokens)
    tokens_arr = tokens_arr[mask]
    tokens = tokens_arr.tolist()
            
    return tokens, mentions, hashtags


def get_remove_links(tokenizer, tokens):
    
    links_strings = []
    encoded_links = []
    
    if special_tokens['https'] in tokens:
        
        decoded_tokens = tokenizer.decode(tokens).split(' ')

        mask = []

        length = len(decoded_tokens)

        finished  = False
        index = 0
        i = 0

        while i < length and not finished:

            dec_t = decoded_tokens[i]

            if dec_t == 'https':
                try:
                    index = i + 7
                    links_strings.append(decoded_tokens[index])
                    for j in range(8):
                        mask.append(False) # link splittato in 8 elementi tutti da rimuovere
                    i += 8
                    #print(initial_index, final_index)
                except:
                    #print(decoded_tokens)
                    #print(i)
                    #print(index)
                    #print(decoded_tokens[i])
                    for j in range(i, length):
                        mask.append(False)
                    finished = True
            else:
                mask.append(True)
                i += 1

        #print(decoded_tokens)
        #print(len(decoded_tokens))
        #print(mask)
        #print(len(mask))

        tokens_arr = np.array(decoded_tokens)
        tokens_arr = tokens_arr[mask]
        decoded_tokens = tokens_arr.tolist()
        tokens = tokenizer.encode(' '.join(decoded_tokens))
        tokens = tokens[1:-1]  # ignore CLS ans SEP (they are duplicated)
    
        #print(tokenizer.decode(tokens))
        # encode only the last string in each link (ignore the "https://t.co")
        for l in range(len(links_strings)):
            if links_strings[l] == '[SEP]':
                links_strings.pop(l)
            else:
                links_strings[l] = links_strings[l].replace(',', '').replace('.','')  # a link cointained ',' (in un csv spostava tutte le colonne)
                enc_l = tokenizer.encode(links_strings[l]) 
                encoded_links.append(enc_l[1:-1])  # ignore CLS ans SEP
        
        #print(links_strings)
    
    else:
        tokens[-1] = '102'  # remove the last "\n" (tokens ends with "102\n")
            
    return tokens, encoded_links, links_strings


def reduce_num_special_tokens(tokens_list):
    for special_token in special_tokens_list:

        if special_token in tokens_list:

            #print('special token: ' + special_token)
            count = 0
            index = 0
            old_token = '101'

            for token in tokens_list:
                if token != old_token and count > 1:
                    #print('index: ' + str(index))
                    #print('count: ' + str(count))
                    #print('current token: ' + token)
                    #print('old token: ' + old_token)
                    tokens_list[index:index+count-1] = [special_token]
                    count = 0
                elif token == special_token:
                    count += 1
                else:
                    index += 1
                    count = 0

                old_token = token
    
    return tokens_list