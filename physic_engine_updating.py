import numpy as np

def get_vec_size(vec):
    return (vec.dot(vec))**0.5

def normalize(vec):
    size = get_vec_size(vec)
    return vec / size

def distance2D(pos1, pos2):
    vec = pos2-pos1
    return (vec.dot(vec))**0.5

class ParticleCrashSimulator:
    def __init__(self, dt=0.1, e=0.5, u=0.4, g=9.8, scene_range=(-50, 50, -50, 50)):
        self.dt = dt    # const, time differential
        self.e = e      # const, repulsion coefficient of A
        self.u = u      # glass on glass friction coefficient
        self.g = g      # m/s^2
        self.limitA = np.array((scene_range[0], scene_range[2]), dtype=np.float32)   #range of scene. no recording after this limit
        self.limitB = np.array((scene_range[1], scene_range[3]), dtype=np.float32)

    def __call__(self, posA, posB, collision_pos, mA, mB, steps1, steps2):
        # posA, posB : input, meter
        # mA, mB : input, kg
        VecA = collision_pos - posA
        VecB = collision_pos - posB

        VecA = normalize(VecA)
        VecB = normalize(VecB)

        disA = distance2D(posA, collision_pos)
        disB = distance2D(posB, collision_pos)

        t = self.dt * steps1

        vA = (self.u * self.g * t) / 2 + disA / t
        vB = (self.u * self.g * t) / 2 + disB / t

        fricA = VecA * self.u * self.g  # accelarate
        fricB = VecB * self.u * self.g  # accelarate

        VA = VecA * vA  # velocity
        VB = VecB * vB  # velocity

        VA_after = VA - fricA*t
        VB_after = VB - fricB*t

        norm_VA_after = normalize(VA_after)
        norm_VB_after = normalize(VB_after)

        if sum(norm_VA_after==VecA) and sum(norm_VB_after==VecB):
            pass
        else:
            return 'impossible case', 0

        Aseq = [posA]
        Bseq = [posB]

        for step in range(1, steps1 + 1):
            xfricA = np.round(normalize(fricA)[0], 5)
            xA = np.round(normalize(VA-fricA*self.dt*step)[0], 5)
            xfricB = np.round(normalize(fricB)[0], 5)
            xB = np.round(normalize(VB-fricB*self.dt*step)[0], 5)

            if xfricA == xA:
                new_posA = posA + (VA * self.dt * step - 0.5 * fricA * (self.dt * step) ** 2)
                Aseq.append(new_posA)
            else:
                Aseq.append(Aseq[-1])

            if xfricB == xB:
                new_posB = posB + (VB * self.dt * step - 0.5 * fricB * (self.dt * step) ** 2)
                Bseq.append(new_posB)
            else:
                Bseq.append(Bseq[-1])

        VA_friced = VA - fricA * t
        VB_friced = VB - fricB * t

        v = VA_friced - VB_friced
        vp = -self.e * v
        dVB = (mA / mB) * (v - vp)
        colli_VB = VB_friced + dVB
        colli_VA = VA_friced + (mB / mA) * (VB_friced - colli_VB)

        colli_fricA = normalize(colli_VA) * self.u * self.g
        colli_fricB = normalize(colli_VB) * self.u * self.g

        for step in range(1, steps2 + 1):
            xfricA = np.round(normalize(colli_fricA)[0], 5)
            xA = np.round(normalize(colli_VA - colli_fricA * self.dt * step)[0], 5)
            xfricB = np.round(normalize(colli_fricB)[0], 5)
            xB = np.round(normalize(colli_VB - colli_fricB * self.dt * step)[0], 5)

            if xfricA == xA:
                new_posA = Aseq[int(steps1)] + (colli_VA * self.dt * step - 0.5 * colli_fricA * (self.dt * step) ** 2)
                limitA_com = new_posA > self.limitA
                limitB_com = new_posA < self.limitB
                limitA_com = limitA_com.astype(np.uint8)
                limitB_com = limitB_com.astype(np.uint8)
                summation = np.sum(limitA_com)+np.sum(limitB_com)
                if summation != 4:
                    break
                Aseq.append(new_posA)
            else:
                Aseq.append(Aseq[-1])

            if xfricB == xB:
                new_posB = Bseq[int(steps1)] + (colli_VB * self.dt * step - 0.5 * colli_fricB * (self.dt * step) ** 2)
                limitA_com = new_posB > self.limitA
                limitB_com = new_posB < self.limitB
                limitA_com = limitA_com.astype(np.uint8)
                limitB_com = limitB_com.astype(np.uint8)
                summation = np.sum(limitA_com) + np.sum(limitB_com)
                if summation != 4:
                    break
                Bseq.append(new_posB)
            else:
                Bseq.append(Bseq[-1])

        Aseq = np.asarray(Aseq)
        Bseq = np.asarray(Bseq)

        arr = [len(Aseq), len(Bseq)]
        smaller = np.argmin(arr)
        smaller_len = arr[smaller]
        Aseq = Aseq[0:smaller_len]
        Bseq = Bseq[0:smaller_len]

        return Aseq, Bseq