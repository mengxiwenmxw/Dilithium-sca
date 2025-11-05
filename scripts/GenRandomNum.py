import random

file_root_path = '/15T/Projects/Dilithium-SCA/data/special_files/' 




def gen_random(file,number,num_arange=8380417,num_start=1):
    with open(file,'w') as of:
        #unique_randoms = random.sample(range(num_start, num_arange), k=number)
        unique_randoms = range(3329+6)
        for num in unique_randoms:
            of.write(str(num)+'\n')
    print(f"--->Gen {number} num in [{num_start},{num_arange-1}], Saved in file: {file}")

if __name__ == "__main__":
    number = 3329
    file_name = f'Random_{number}.txt'
    file = file_root_path + file_name
    gen_random(file=file, number=number)