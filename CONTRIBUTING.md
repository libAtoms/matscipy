Contributing to matscipy
========================

Code style
----------
Always follow [PEP-8](https://www.python.org/dev/peps/pep-0008/), with the following exception: "One big exception to PEP 8 is our preference of longer line lengths. We’re well into the 21st Century, and we have high-resolution computer screens that can fit way more than 79 characters on a screen. Don’t limit lines of code to 79 characters if it means the code looks significantly uglier or is harder to read." (Taken from [Django's contribuing guidelines](https://docs.djangoproject.com/en/dev/internals/contributing/writing-code/coding-style/).)

Development branches
--------------------
New features should be developed always in its own branch. When creating your own branch,
please suffix that branch by the year of creation on a description of what is contains.
For example, if you are working on an implementation for line scans and you started that
work in 2018, the branch could be called "18_line_scans".

Commits
-------
Prepend you commits with a shortcut indicating the type of changes they contain:
* API: changes to the user exposed API
* BUG: Bug fix
* BUILD: Changes to the build system
* CI: Changes to the CI configuration
* DOC: Changes to documentation strings or documentation in general (not only typos)
* ENH: Enhancement (e.g. a new feature)
* MAINT: Maintenance (e.g. fixing a typo, or changing code without affecting function)
* TST: Changes to the unit test environment
* WIP: Work in progress

The changelog will be based on the content of the commits with tag BUG, API and ENH.

Examples: 
- If your are working on a new feature, use ENH on the commit making the feature ready. Before use the WIP tag.
- use TST when your changes only deal with the testing environment. If you fix a bug and implement the test for it, use BUG.
- minor changes that doesn't change the codes behaviour (for example rewrite file in a cleaner or slightly efficienter way) belong to the tag MAINT
- if you change documentation files without changing the code, use DOC; if you also change code in the same commit, use another shortcut
