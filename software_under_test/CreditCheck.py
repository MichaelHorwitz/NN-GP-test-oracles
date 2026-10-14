from dataclasses import dataclass
import random
from typing import List, Set

import pandas as pd
from keras.utils import to_categorical
from generation_methods.LinearGeneration import LinearGeneration
from generation_methods.RandomSampler import RandomSampler
from gp.Node import Node
from software_under_test.SoftwareUnderTest import SoftwareUnderTest

@dataclass
class Applicant:
    id: int = 0 # >= 0
    citizenShip: int = 0 # 0 or 1
    state: int = 0 # 0 or 1
    age: int = 0 # 1-100
    sex: int = 0 # 0 or 1
    region: int = 0 # 0-6
    incomeClass: int = 0 # 0-3
    numberDependents: int = 0 # 0-4
    maritalStatus: int = 0 # 0 or 1
    creditApproved: bool = False 
    creditAmount: float = 0     
    
class CreditCheck(SoftwareUnderTest):
    def __init__(self):
        super().__init__()
        self.number_branches = 23
        self.__name__ = "Credit"
    def generate_test_from_tree(self, tree: Node) -> Applicant:
        # Helper to safely convert tree output to int for modulo operations
        def _safe_int():
            try:
                val = tree.do_compute()
                return int(val)
            except (ValueError, OverflowError):
                return 0
        
        return Applicant(
            id = random.randint(0, 10000000),
            citizenShip=_safe_int() % 2,
            state=_safe_int() % 2,
            age=(_safe_int() % 100) + 1,
            sex=_safe_int() % 2,
            region=_safe_int() % 7,
            incomeClass=_safe_int() % 4,
            numberDependents=_safe_int() % 5,
            maritalStatus=_safe_int() % 2,
            creditApproved=False,
            creditAmount  =0
        )
    def generate_test_from_random(self, rng: RandomSampler):
        return Applicant(
            id=rng.generate_int(0, 10000000),
            citizenShip=rng.generate_int(0, 2),
            state=rng.generate_int(0, 2),
            age=rng.generate_int(1, 101),
            sex=rng.generate_int(0, 2),
            region=rng.generate_int(0, 7),
            incomeClass=rng.generate_int(0, 4),
            numberDependents=rng.generate_int(0, 5),
            maritalStatus=rng.generate_int(0, 2),
            creditApproved=False,
            creditAmount=0
        )
    def generate_test_from_linear(self, linear: LinearGeneration) -> Applicant:
        app_tuple = linear.generate_applicant_tuple()
        return Applicant(
            id=int(linear.curr_num),
            citizenShip=app_tuple[0],
            state=app_tuple[1],
            age=app_tuple[2],
            sex=app_tuple[3],
            region=app_tuple[4],
            incomeClass=app_tuple[5],
            numberDependents=app_tuple[6],
            maritalStatus=app_tuple[7],
            creditApproved=False,
            creditAmount=0
        )
    def find_coverage(self, test_suite) -> float:
        set_of_branches: Set[str] = set()
        for test in test_suite:
            list_of_branches = self.exec_sut_coverage(test)
            for branch in list_of_branches:
                set_of_branches.add(branch)
        return len(set_of_branches) / self.number_branches
    
    def program_input_to_pandas(self, inputs: list[Applicant]) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(inputs)
        features = df.drop(columns=['creditApproved']).astype('float32')
        labels = pd.DataFrame(to_categorical(df["creditApproved"].astype(int), num_classes=2)).astype('float32')
        return (features, labels)
    
    def exec_sut_coverage(self, user: Applicant) -> List[str]:
        return_set: List[str] = []
        creditLimit: float = 0
        if 5 == user.region or 6 == user.region:
            return_set.append("red")
            creditLimit = 0
        else:
            return_set.append("blue")
            if user.age < 18:
                return_set.append("yellow")
                creditLimit = 0
            elif user.citizenShip == 0:
                return_set.append("green")
                creditLimit = 5000 + 1000 * user.incomeClass
                if user.state == 0:
                    return_set.append("purple")
                    if user.region == 3 or user.region == 4:
                        return_set.append("orange")
                        creditLimit *= 2
                    else:
                        return_set.append("pink")
                        creditLimit *= 1.5
                else:
                    return_set.append("cyan")
                    creditLimit *= 1.1
                if user.maritalStatus == 0:
                    return_set.append("brown")
                    if user.numberDependents > 0:
                        return_set.append("gray")
                        creditLimit += 200*user.numberDependents
                    else:
                        return_set.append("lime")
                        creditLimit += 500
                else:
                    return_set.append("navy")
                    creditLimit += 1000
                if user.sex == 0:
                    return_set.append("maroon")
                    creditLimit += 500
                else:
                    return_set.append("olive")
                    creditLimit += 1000
            else:
                return_set.append("teal")
                creditLimit = 1000 + 800 * user.incomeClass
                if user.maritalStatus == 0:
                    return_set.append("silver")
                    if user.numberDependents > 2:
                        return_set.append("gold")
                        creditLimit += 100 * user.numberDependents
                    else:
                        return_set.append("indigo")
                        creditLimit += 100
                else:
                    return_set.append("violet")
                    creditLimit += 300
                if user.sex == 0:
                    return_set.append("coral")
                    creditLimit += 100
                else:
                    return_set.append("khaki")
                    creditLimit += 200

        user.creditAmount = int(creditLimit)

        # Branch 11: Final approval check
        if user.creditAmount == 0:
            return_set.append("black")  # Branch 11a: Not approved
            user.creditApproved = False
        else:
            return_set.append("white")  # Branch 11b: Approved
            user.creditApproved = True

        return return_set
    def exec_real_output(self, user: Applicant) -> bool:
        creditLimit = 0
        if 5 == user.region or 6 == user.region:
            creditLimit = 0
        else:
            if user.age < 18:
                creditLimit = 0
            elif user.citizenShip == 0:
                creditLimit = 5000 + 1000 * user.incomeClass
                if user.state == 0:
                    if user.region == 3 or user.region == 4:
                        creditLimit *= 2
                    else:
                        creditLimit *= 1.5
                else:
                    creditLimit *= 1.1
                if user.maritalStatus == 0:
                    if user.numberDependents > 0:
                        creditLimit += 200*user.numberDependents
                    else:
                        creditLimit += 500
                else:
                    creditLimit += 1000
                if user.sex == 0:
                    creditLimit += 500
                else:
                    creditLimit += 1000
            else:
                creditLimit = 1000 + 800 * user.incomeClass
                if user.maritalStatus == 0:
                    if user.numberDependents > 2:
                        creditLimit += 100 * user.numberDependents
                    else:
                        creditLimit += 100
                else:
                    creditLimit += 300
                if user.sex == 0:
                    creditLimit += 100
                else:
                    creditLimit += 200
        user.creditAmount = creditLimit
        if user.creditAmount == 0:
            user.creditApproved = False
            return user.creditApproved
        else:
            user.creditApproved = True
            return user.creditApproved
    def get_num_outputs(self) -> int:
        return 2
    def get_tailored_suite(self) -> tuple[List[Applicant], List[bool]]:
        # Archetypes define the minimal set of attributes to trigger a specific path.
        path_archetypes = [
            # Path 1: Red
            {'region': 5},
            # Path 2: Yellow
            {'age': 17},

            # Paths 3-20: Green (18 paths)
            # Base: age >= 18, citizenShip = 0
            # Combinations of:
            # - state/region -> purple(orange/pink) | cyan
            # - maritalStatus/dependents -> brown(gray/lime) | navy
            # - sex -> maroon | olive
            {'state': 0, 'region': 3, 'maritalStatus': 0, 'numberDependents': 1, 'sex': 0}, # orange, gray, maroon
            {'state': 0, 'region': 3, 'maritalStatus': 0, 'numberDependents': 1, 'sex': 1}, # orange, gray, olive
            {'state': 0, 'region': 3, 'maritalStatus': 0, 'numberDependents': 0, 'sex': 0}, # orange, lime, maroon
            {'state': 0, 'region': 3, 'maritalStatus': 0, 'numberDependents': 0, 'sex': 1}, # orange, lime, olive
            {'state': 0, 'region': 3, 'maritalStatus': 1, 'sex': 0},                      # orange, navy, maroon
            {'state': 0, 'region': 3, 'maritalStatus': 1, 'sex': 1},                      # orange, navy, olive
            {'state': 0, 'region': 1, 'maritalStatus': 0, 'numberDependents': 1, 'sex': 0}, # pink, gray, maroon
            {'state': 0, 'region': 1, 'maritalStatus': 0, 'numberDependents': 1, 'sex': 1}, # pink, gray, olive
            {'state': 0, 'region': 1, 'maritalStatus': 0, 'numberDependents': 0, 'sex': 0}, # pink, lime, maroon
            {'state': 0, 'region': 1, 'maritalStatus': 0, 'numberDependents': 0, 'sex': 1}, # pink, lime, olive
            {'state': 0, 'region': 1, 'maritalStatus': 1, 'sex': 0},                      # pink, navy, maroon
            {'state': 0, 'region': 1, 'maritalStatus': 1, 'sex': 1},                      # pink, navy, olive
            {'state': 1, 'maritalStatus': 0, 'numberDependents': 1, 'sex': 0},             # cyan, gray, maroon
            {'state': 1, 'maritalStatus': 0, 'numberDependents': 1, 'sex': 1},             # cyan, gray, olive
            {'state': 1, 'maritalStatus': 0, 'numberDependents': 0, 'sex': 0},             # cyan, lime, maroon
            {'state': 1, 'maritalStatus': 0, 'numberDependents': 0, 'sex': 1},             # cyan, lime, olive
            {'state': 1, 'maritalStatus': 1, 'sex': 0},                                   # cyan, navy, maroon
            {'state': 1, 'maritalStatus': 1, 'sex': 1},                                   # cyan, navy, olive

            # Paths 21-26: Teal (6 paths)
            # Base: age >= 18, citizenShip != 0
            # Combinations of:
            # - maritalStatus/dependents -> silver(gold/indigo) | violet
            # - sex -> coral | khaki
            {'citizenShip': 1, 'maritalStatus': 0, 'numberDependents': 3, 'sex': 0}, # silver, gold, coral
            {'citizenShip': 1, 'maritalStatus': 0, 'numberDependents': 3, 'sex': 1}, # silver, gold, khaki
            {'citizenShip': 1, 'maritalStatus': 0, 'numberDependents': 2, 'sex': 0}, # silver, indigo, coral
            {'citizenShip': 1, 'maritalStatus': 0, 'numberDependents': 2, 'sex': 1}, # silver, indigo, khaki
            {'citizenShip': 1, 'maritalStatus': 1, 'sex': 0},                      # violet, coral
            {'citizenShip': 1, 'maritalStatus': 1, 'sex': 1},                      # violet, khaki
        ]

        applicants: List[Applicant] = []

        # Generate 100 applicants by cycling through the archetypes.
        # This creates 4 instances for the first 22 archetypes and 3 for the last 4.
        # (22 * 4) + (4 * 3) = 100
        for i in range(100):
            archetype = path_archetypes[i % len(path_archetypes)]

            # Set default values that pass the initial checks
            base_params = {
                'region': 1,
                'age': 30,
                'citizenShip': 0,
                'state': 0,
                'incomeClass': 2,
                'maritalStatus': 0,
                'numberDependents': 0,
                'sex': 0,
            }

            # Create the final parameters by overriding defaults with the archetype's values
            final_params = {**base_params, **archetype}
            applicants.append(Applicant(**final_params, creditAmount=0, creditApproved=False))
        output = [self.exec_real_output(user) for user in applicants]
        return (applicants, output)