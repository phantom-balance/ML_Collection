# class B():
#
#     def __init__(self,ram):
#         self.ram=ram
#         self.rampow2=ram**2
#     def hot(self):
#         return(0)
#     @classmethod
#     def print_oice(cls, pam):
#         print(f"noice{cls.hot}{pam}")
#
#     @staticmethod
#     def print_noice(pam):
#         print(f"noice{pam+1}")
# class A():
#     def __init__(self,ram):
#         self.ram=ram
#         self.other = B(ram)
#         print(self.other.rampow2)
#
# A(3)
# B.print_noice(100)
# B.print_oice(100)
# kam={0:"hot", 1:"not"}
# print(kam[0])
#################################################################
#                                                               #
#################################################################
# import spacy
# spacy_eng = spacy.blank("en")
# a= [tok.text.lower() for tok in spacy_eng.tokenizer("night will fight")]
# print(a)
#################################################################
#                                                               #
#################################################################
# import torch
# batch_size=3
# a=torch.rand(batch_size,3,4,4)
# print(a.shape)
# print(a.unsqueeze(0).shape) # Not correct
# print(a[0].unsqueeze(0).shape)
# print(a[0].shape)
# print(a[0].unsqueeze(0).shape for a in a)

# a=torch.rand(2,3,2)
# print(a)
# print(a.shape)
# print(a.reshape(-1, a.shape[2]))
# print(a.shape)
# print(a.argmax(0))
# print(a.argmax(1))
# print(a.argmax(2))
# print(a.unsqueeze(0))
# print(a.unsqueeze(0).shape)
# print(a.unsqueeze(0).argmax(1))
# b= torch.tensor([1,2,3])
# print(b.argmax(0).item())
#################################################################
#                                                               #
#################################################################

