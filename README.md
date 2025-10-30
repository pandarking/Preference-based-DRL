# Preference-based Deep Reinforcement Learning for Historical Route Estimation
Our paper [Preference-based Deep Reinforcement Learning for Historical Route Estimation](https://www.ijcai.org/proceedings/2025/0955.pdf) has been accepted to IJCAI 2025.

We propose a preference-based DRL method characterized by its reward design and optimization objective, which is specialized to learn historical route preferences. Our experiments demonstrate that the method aligns generated solutions more closely with human preferences. Moreover, it exhibits strong generalization performance across a variety of instances, offering a robust solution for different VRP scenarios.

### Training
```python
python train_n20.py
```
### Testing
```python
python test_n20.py
```
***
## Reference
If you find this codebase useful, please consider citing the paper:
```latex
@inproceedings{ijcai2025p955,
  title     = {Preference-based Deep Reinforcement Learning for Historical Route Estimation},
  author    = {Pan, Boshen and Wu, Yaoxin and Cao, Zhiguang and Hou, Yaqing and Zou, Guangyu and Zhang, Qiang},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {8591--8599},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/955},
  url       = {https://doi.org/10.24963/ijcai.2025/955},
}
```
