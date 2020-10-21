import numpy as np
from math import log
from anytree import Node

"""
ID3 algorithm steps:
1. Create the root node of the tree T
2. If all the examples in S are of the same class c, returns the tree T labeled with class c
3. If A is empty, returns the tree T labeled with the majority class c in S
4. Let a belongs to A such that a is optimal in A
5. Partition the set S according to the possible values that the optimal attribute a can assume
6. Recursive call of ID3 --> ID3(S_(a=v_j), A - a)
"""


class ID3:

    # Step 1: create the root node
    T = Node("Root")

    def __init__(self, S, A):
        self.algorithm(S, A, self.T)

    def algorithm(self, S, A, T):
        # Step 2: if all the examples in S are of the same class c, returns the tree T labeled with class c
        c = self.areAllElementOfSetEqual(S)
        if c != "":
            return Node(c, parent=T)

        # Step 3: if A is empty, returns the tree T labeled with the majority class c in S
        if not A:
            c = self.majorityClassOfSet(S, A)
            return Node(c, parent=T)

        # Step 4: let a belongs to A such that a is optimal in A
        a = self.optimalAttribute(S, A)

        # Get all the values that the optimal attribute a can assume in S
        values = self.valuesByAttribute(S, a)

        # Update the tree T
        T_prime = Node(a, parent=T, value=values)

        # Remove the current optimal attribute a from A
        A.remove(a)

        # Make a recursive call for each value that the optimal attribute a can assume in S
        for i in range(len(values)):

            # Step 5: partition the set S according to the possible values that the optimal attribute a can assume
            S_prime = self.partition(S, a, values[i])

            # Step 6: recursive call of ID3
            self.algorithm(S_prime, A, T_prime)

    # Check if all the elements of a set are of the same class c
    def areAllElementOfSetEqual(self, S):
        c = S[0]["Sport"]
        for i in range(1, self.cardinality(S)):
            if S[i]["Sport"] != c:
                return ""
        return c

    # Determine the majority class in S
    def majorityClassOfSet(self, S, A):
        classes = {}

        for s in S:
            if s["Sport"] not in classes:
                classes[s["Sport"]] = 1
            else:
                classes[s["Sport"]] += 1

        return A[int(np.argmax(classes))]

    # Determine the optimal attribute in A
    def optimalAttribute(self, S, A):

        # If the |A| = 1, consider the only one attribute in A as optimal
        if self.cardinality(A) == 1:
            return A[0]

        information_gains = []

        for a in A:
            information_gains.append(self.informationGain(S, a))

        # Determine which attribute has the highest Information Gain
        index = np.argmax(information_gains)

        return A[int(index)]

    # Get the cardinality of a set
    def cardinality(self, S):
        return len(S)

    # Calculate the Information Gain
    def informationGain(self, S, x):
        values = {}
        summation = 0

        for s in S:
            if s[x] not in values:
                values[s[x]] = 1
            else:
                values[s[x]] += 1

        for v in values:
            # Get the examples from S by the value v of the attribute x
            s_x = self.examplesByAttribute(S, x, v)

            summation += (values[v] / self.cardinality(S)) * self.entropy(s_x, method="cross-entropy")

        return self.entropy(S, method="cross-entropy") - summation

    # Get the examples from the set S with attribute x and value v
    def examplesByAttribute(self, S, x, v):
        s_x = []
        for s in S:
            if s[x] == v:
                s_x.append(s)
        return s_x

    # Calculate the entropy
    def entropy(self, S, method):
        if method == "cross-entropy":
            return self.crossEntropy(S)
        if method == "gini-impurity":
            return self.giniImpurity(S)

    # Calculate the Cross-Entropy
    def crossEntropy(self, S):
        classes = {}

        for s in S:
            if s["Sport"] not in classes:
                classes[s["Sport"]] = 1
            else:
                classes[s["Sport"]] += 1

        E = 0
        for c in classes:
            p_c = classes[c] / self.cardinality(S)
            E += p_c * log(p_c, 2)

        return -E

    # Calculate the Gini Impurity
    def giniImpurity(self, S):
        classes = {}

        for s in S:
            if s["Sport"] not in classes:
                classes[s["Sport"]] = 1
            else:
                classes[s["Sport"]] += 1

        GI = 0
        for c in classes:
            p_c = classes[c] / self.cardinality(S)
            GI += p_c * p_c

        return 1 - GI

    # Get the values that an attribute x can assume in S
    def valuesByAttribute(self, S, x):
        values = []

        for i in range(self.cardinality(S)):
            if S[i][x] not in values:
                values.append(S[i][x])

        return values

    # Partition the set S by the value v that an attribute x can assume in S
    def partition(self, S, x, v):
        partitions = []

        for i in range(self.cardinality(S)):
            if S[i][x] == v:
                partitions.append(S[i])

        return partitions
