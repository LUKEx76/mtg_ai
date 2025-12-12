import hashlib

# NO SUCCESS SO FAR...
def identify_hash_function(possible_inputs, target_hash):
    clean_target = target_hash.strip()
    
    algorithms_to_check = [
        'md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512',
        'blake2b', 'blake2s', 'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'ripemd160', 'whirlpool'
    ]
    
    encodings_to_try = ['utf-8', 'ascii', 'latin-1']

    print(f"--- Analyzing Hash: {clean_target} ---")
    for text_input in possible_inputs:
        print(f"Checking input: '{text_input}'")
        
        for encoding in encodings_to_try:
            try:
                input_bytes = text_input.encode(encoding)
            except UnicodeEncodeError:
                continue

            for algo_name in algorithms_to_check:
                if algo_name not in hashlib.algorithms_available:
                    continue

                try:
                    h = hashlib.new(algo_name)
                    h.update(input_bytes)
                    digest = h.hexdigest()

                    if digest.lower() == clean_target.lower():
                        case_match = "Exact" if digest == clean_target else "Case-Insensitive (Value Match)"
                        
                        return {
                            "found": True,
                            "algorithm": algo_name,
                            "encoding": encoding,
                            "matching_input": text_input,
                            "generated_hash": digest,
                            "match_type": case_match
                        }
                except Exception as e:
                    pass

    return {"found": False}

# the MuZ2MLc8SEtUXZB9xfwdh seems to be the hash of the commanders name Avacyn, Angel of Hope

inputs = ["317f1133-7cf8-4b7a-919e-88c45f8c2c3a", "Avacyn, Angel of Hope", "Avacyn, Angel of Hope".lower(), "avacyn-angel-of-hope", "avacyn-angel-of-hope".replace("-", " "), "avacyn-angel-of-hope".replace("-", "")]
target_output = "MuZ2MLc8SEtUXZB9xfwdh"

result = identify_hash_function(inputs, target_output)

print("\n" + "="*30)
if result['found']:
    print(f"SUCCESS! Match Found.")
    print(f"Algorithm:   {result['algorithm']}")
    print(f"Encoding:    {result['encoding']}")
    print(f"Input:       '{result['matching_input']}'")
    print(f"Generated:   {result['generated_hash']}")
    print(f"Target:      {target_output}")
    print(f"Match Type:  {result['match_type']}")
else:
    print("FAILURE. No match found.")
print("="*30)