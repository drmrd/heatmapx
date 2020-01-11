import sys

import git


sys.path.append(git.Repo(search_parent_directories=True).working_tree_dir)
