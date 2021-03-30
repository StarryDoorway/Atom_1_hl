# Atom_1_hl
Look for the stable structure of Silicon atom
python 3.8
tensorflow 2.3
gym 0.18.0
记得注册环境atom.py


    Description:
        To find the structrue of low cost;
        3 atoms for now(num_atoms = 3);
        The tool to culculate the cost is provided by  College of Chemistry , Jilin University.

    Observation:
        Type: Discrete(3 * num_atoms)
        Num	Observation               Min             Max
        0	x of atom_1                0               1
        1	y of atom_1                0               1
        2	z of atom_1                0               1
        3	x of atom_2                0               1
        ...
        8   z of atom_3                0               1



    Actions:
        Type: Discrete(6 * num_atoms)
        Num	 Action
        0	 x of atom_1  +0.01
        1	 y of atom_1  +0.01
        2	 z of atom_1  +0.01
        3	 x of atom_2  +0.01
        ...
        17   z of atom_3  -0.01


        Note: action 0~5  belongs to atom_1;6~11 belongs to atom_2 ... They all +0.01.
              action 9~11 belongs to atom_1 ... 15~17 belongs to atom_3.They all -0.01


    Reward:
        Reward is -1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [0,1]

    Episode Termination:
        1.Episode length is greater than 5000.
        2.Have found the structure we need.
