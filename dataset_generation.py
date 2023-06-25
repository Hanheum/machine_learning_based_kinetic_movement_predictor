import numpy as np
from physic_engine_updating import ParticleCrashSimulator
import pandas as pd
import random
from os.path import isfile

phys_en = ParticleCrashSimulator(dt=0.1, e=0.1)

def particle_crash_simulation(posA, posB, collide_position, mA, mB, step1, step2):
    Aseq, Bseq = phys_en(posA=posA, posB=posB, collision_pos=collide_position, mA=mA, mB=mB, steps1=step1, steps2=step2)
    if np.sum(Bseq) != 0:
        combined = np.concatenate((Aseq.T, Bseq.T), axis=0)
        mA_arr = np.ones((1, Aseq.shape[0])) * mA
        mB_arr = np.ones((1, Aseq.shape[0])) * mB
        combined = np.concatenate((combined, mA_arr), axis=0)
        combined = np.concatenate((combined, mB_arr), axis=0)
        combined = combined.T
        return combined
    else:
        return 0

def generate_random_particle():
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    return np.array((x, y))

def generate_random_particle_dis(A, r):
    which = [0, 1]
    which = random.choice(which)
    a, b = A
    Bx = random.uniform(a-r, a+r)

    if which:
        y = b + (r**2 - (Bx-a)**2)**0.5
    else:
        y = b - (r**2 - (Bx-a)**2)**0.5

    return np.array((Bx, y))

def generate_radiuses():
    rB = random.uniform(80, 100)
    rC = random.uniform(rB/2, rB/2 + 1)
    return rB, rC

def generate_masses():
    mA = random.uniform(2, 3)
    mB = random.uniform(2, 3)
    return mA, mB

def generate_steps():
    step1 = random.randint(20, 30)
    step2 = random.randint(20, 30)
    return step1, step2

def generate_data(event_num):
    rB, rC = generate_radiuses()
    A = generate_random_particle()
    B = generate_random_particle_dis(A, rB)
    collision = generate_random_particle_dis(A, rC)
    mA, mB = generate_masses()
    step1, step2 = generate_steps()

    data = particle_crash_simulation(A, B, collision, mA, mB, step1, step2)
    try:
        if data == 0:
            return 0
    except:
        pass
    event_arr = np.ones((1, data.shape[0]))*event_num
    data = np.concatenate((event_arr, data.T), axis=0).T
    data = pd.DataFrame(data, columns=['event_num', 'Ax', 'Ay', 'Bx', 'By', 'mA', 'mB'])

    return data

saving_dir = 'C:\\Users\\chh36\\Desktop\\data.csv'
start_num = int(input('event num:'))
target_num = int(input('target num:'))
while start_num != target_num:
    new_df = generate_data(start_num)
    try:
        if new_df == 0:
            continue
    except:
        pass
    if isfile(saving_dir):
        df = pd.read_csv(saving_dir)
        df = pd.concat([df, new_df])
        df.to_csv(saving_dir, index=False)
    else:
        new_df.to_csv(saving_dir, index=False)
    start_num += 1

    progress_rate = start_num/target_num
    percentage = f'{round(progress_rate*100)}%'
    equal_sign = round(50*progress_rate)
    space_sign = 50-equal_sign
    txt = '[{}{}], {}'.format('='*equal_sign, ' '*space_sign, percentage)
    print(txt, end='\r')