Conclusion with Domain Classifier Experiments...


Below are the points:-
1) The domain classifier loss is initially high but converges quickly to reduce the loss in domain classification, if shift is present
2) The test accuracy of Digit Classification is generally poor and it doesn't increase much when using a basic simple implementation of Domain Classifier, if shift is present. 
3) During the training process, the domain classifier achieves high accuracies early in the epochs if shift present. 
4) If domain Shift is absent, then test accuracies of digit classification is high, since both are from same domain
5) If domain shift is absent, then domain classifier loss doesn't converge and remains high even at end of epoch
6) If domain shift is absent, the domain classifier accuracy oscillates somewhere in the vicinirty of 50 percent.


So summarizing, 
Higher domain classification accuracies, Relatively lower test accuracies of Digit Classification and Converging Loss of Domain classifier conveys domain shift present.
Lower domain classification accuracies (around 50%), High test accuracies of Digit Classification and Non-Convergance Loss of Domain classifier conveys domain shift absent.
