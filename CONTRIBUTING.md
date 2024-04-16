<!-- omit in toc -->
# Contributing to AIgarMIC

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions.

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - [Cite](https://github.com/agerada/AIgarMIC?tab=readme-ov-file#cite) the project or associated validation work
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->
## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving The Documentation](#improving-the-documentation)
- [Styleguides](#styleguides)
  - [Commit Messages](#commit-messages)
- [Join The Project Team](#join-the-project-team)


## Code of Conduct

This project and everyone participating in it is governed by the
[AIgarMIC Code of Conduct](https://github.com/agerada/AIgarMICblob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to <alessandro.gerada@liverpool.ac.uk>.


## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://aigarmic.readthedocs.io/en/latest/).

Before you ask a question, it is best to search for existing [Issues](https://github.com/agerada/AIgarMIC/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Start a topic in the [Discussions](https://github.com/agerada/AIgarMIC/discussions) section (alternatively, email the lead developer at <alessandro.gerada@liverpool.ac.uk>).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (especially `tensorflow` and `keras` versions), depending on what seems relevant.

We will then take care of the issue as soon as possible.


## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs

<!-- omit in toc -->
#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g., using incompatible environment components/versions (Make sure that you have read the [documentation](https://aigarmic.readthedocs.io/en/latest/). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/agerada/AIgarMICissues?q=label%3Abug).
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of `python`, `tensorflow` and `keras`
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

<!-- omit in toc -->
#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead, sensitive bugs must be sent by email to <alessandro.gerada@liverpool.ac.uk>.
<!-- You may add a PGP key to allow the messages to be sent encrypted as well. -->

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/agerada/AIgarMIC/issues/new). (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as well as possibly other tags (such as `critical`), and the issue will be left to be [implemented by someone](#your-first-code-contribution).

<!-- You might want to create an issue template for bugs and errors that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for AIgarMIC, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

<!-- omit in toc -->
#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://aigarmic.readthedocs.io/en/latest/) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/agerada/AIgarMIC/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.
- Consider whether the enhancement can be satisfied by using alternative software, such as [`cellprofiler`](https://github.com/CellProfiler/CellProfiler) (for agar plate image manipulation), or [`AMR`](https://github.com/msberends/AMR) (for analysis and processing of minimum inhibitory concentration results). Generally, `AIgarMIC` enhancements that are alternatively feasible using such software will only be considered if they have a significant positive impact on minimum inhibitory concentration analysis routine workflows.

<!-- omit in toc -->
#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/agerada/AIgarMIC/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux. <!-- this should only be included if the project has a GUI -->
- **Explain why this enhancement would be useful** to most AIgarMIC users. You may also want to point out the other projects that solved it better and which could serve as inspiration.

<!-- You might want to create an issue template for enhancement suggestions that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->

### Your First Code Contribution
<!-- TODO
include Setup of env, IDE and typical getting started instructions?

-->

[Installation](https://aigarmic.readthedocs.io/en/latest/installation.html) instructions and [developer notes](https://aigarmic.readthedocs.io/en/latest/developer.html) for `AIgarMIC` can be found in the documentation. We suggest using [PyCharm](https://www.jetbrains.com/pycharm/) as the IDE for development, although this is optional. For development purposes, we recommend:

- Use of a virtual environment
- Installation of optional asset files (https://aigarmic.readthedocs.io/en/latest/installation.html#assets-installation)

Contribute to `AIgarMIC` by following these general steps:

1. Fork the project locally,
2. Create a new branch that will contain your feature/bug fix,
3. Make your updates,
4. If the branch contains new feature/s functionality, add testing code to the `tests` directory (we use `pytest` as the testing framework),
5. Ensure that the full testing suite passes (including ones requiring optional assets) -- see [developer notes]((https://aigarmic.readthedocs.io/en/latest/developer.html)) for more information on testing,
6. Check that build process succeeds (we use `poetry` for dependency management and building), again see [developer notes](https://aigarmic.readthedocs.io/en/latest/developer.html) for more information,
7. Update documentation (if appropriate) and check that `doctest` passes,
8. Commit your changes and push to your fork,
9. Create a pull request to the `main` branch of the `AIgarMIC` repository.


### Improving The Documentation
<!-- TODO
Updating, improving and correcting the documentation

-->

For documentation, we use `reStructuredText`. Source files are stored in the `docs` directory. The documentation is built using `sphinx`. Documentation source code can be updated using the same contribution process as for code. Please try to keep the documentation free of excessive laboratory jargon, and provide examples using the optional assets wherever possible.

## Styleguides

`AIgarMIC` follows the [PEP 8](https://pep8.org/) style guide, although convention and refactor recommendations are not enforced. Please see developer notes for more information on checking `PEP 8` compliance.

In addition, these are some specific style guides for `AIgarMIC`:

- User-accessible or API-level classes and functions/methods should have a `sphinx` compliant `reStructuredText` docstring.
- Comments should be used sparingly
- Maximum line length is 120 characters
- Use of type hints in functions/methods
- Import and no-member errors related to `tensorflow`, `keras` and `openCV` are known and can be ignored inline using: `# pylint: disable=import-error,no-member`


## Join The Project Team and Collaborations

If you are interested in joining the team or a formal collaboration, please email the lead developer, Alessandro Gerada at: <alessandro.gerada@liverpool.ac.uk>.

<!-- omit in toc -->
## Attribution
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
