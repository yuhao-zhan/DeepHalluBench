from nltk.chunk import RegexpParser
from nltk import word_tokenize, pos_tag
import re
import json


def clean_name(name):
    """
    Removes punctuation from a name while preserving spaces between words.
    Filters out words that are all capital letters.
    """
    # Remove punctuation but keep spaces between words
    cleaned = re.sub(r'[^\w\s]', '', name)
    # Split into words and filter out all-caps words
    words = cleaned.split()
    filtered_words = []
    for word in words:
        # Keep words that are not obvious acronyms, e.g., APOE4 should be filtered out
        # Filter out words that are all uppercase and contain only letters and numbers (like APOE4)
        if not (len(word) <= 5 and word.isupper() and word.isalnum()):
            filtered_words.append(word)
    # Join the remaining words and normalize whitespace
    cleaned = ' '.join(filtered_words)
    return cleaned


def sub_leaves(tree, label):
    """
    Extracts the leaves (words) from subtrees with a specific label.
    """
    leaves = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == label):
        leaf_list = subtree.leaves()
        if leaf_list and isinstance(leaf_list[0], str):
            leaves.append(" ".join(leaf_list))
        elif leaf_list:
            # Extract just the words, not the POS tags
            words = [leaf[0] if isinstance(leaf, tuple) else str(leaf) for leaf in leaf_list]
            leaves.append(" ".join(words))
    return leaves


def extract_proper_names(text):
    """
    Extracts proper names from a text using NLTK's RegexpParser.
    First removes punctuation from the input text, then processes it.
    """
    # Remove punctuation from the input text first
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Tokenize and POS tag the cleaned text
    tokens = word_tokenize(cleaned_text)
    tagged_tokens = pos_tag(tokens)
    
    chunk_parser = RegexpParser(r"""
        NP: {<NNP>+}  # Proper nouns (NNP)
        """)
    
    # Parse the tagged tokens from the cleaned text
    tree = chunk_parser.parse(tagged_tokens)
    raw_names = sub_leaves(tree, 'NP')
    
    # Clean the extracted names by removing punctuation
    cleaned_names = [clean_name(name) for name in raw_names]

    # If the name is the first word of the text, remove it
    # if cleaned_names and cleaned_names[0] == text.split()[0]:
    #     cleaned_names = cleaned_names[1:]

    return cleaned_names

if __name__ == "__main__":
    text = "Five open positions at OpenAI align with the user's preferences."
    proper_names = extract_proper_names(text)
    for name in proper_names:
        print(name)

    web_content = """
    "Title: DeepMind\n\nURL Source: https://boards.greenhouse.io/deepmind/jobs/6541427\n\nMarkdown Content:\nJobs at DeepMind\n\n===============\n\nCurrent openings at DeepMind\n============================\n\nCreate a Job Alert\n\nLevel-up your career by having opportunities at DeepMind sent directly to your inbox.\n\nCreate alert\n\nSearch \n\n83 jobs\n-------\n\n### Central Operations, Responsibility, and Engagement\n\n| Job |\n| --- |\n| [Administrative Business Partner Tokyo, Japan](https://job-boards.greenhouse.io/deepmind/jobs/6890209) |\n| [Administrative Business Partner New Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7145029) |\n| [Analytics Engineer Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/6383777) |\n| [Director, Google DeepMind Impact Accelerator Singapore](https://job-boards.greenhouse.io/deepmind/jobs/7126983) |\n| [E/ABP Community Partner Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7074559) |\n| [Events Manager ( 9 Month Fixed Term Contract )New Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7144763) |\n| [Executive Communications Lead, Office of the Chief Operating Officer (OCOO) London, UK](https://job-boards.greenhouse.io/deepmind/jobs/6986419) |\n| [Executive Communications Manager London, UK](https://job-boards.greenhouse.io/deepmind/jobs/6750636) |\n| [Executive Producer Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7103989) |\n| [Head of Frontier Policy Partnerships London, UK](https://job-boards.greenhouse.io/deepmind/jobs/7050399) |\n| [Head of Legal Operations Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/6922383) |\n| [Internal Communications - Unit Head, GenAI Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/6750311) |\n| [Lead Technical Program Manager, Science & Strategic Initiatives London, UK](https://job-boards.greenhouse.io/deepmind/jobs/6975896) |\n| [Legal Counsel - Product Counsel Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7014446) |\n| [P&C Partnering Lead ( 12 Month Fixed Term Contract )New Mountain View, California, US; New York City, New York, US](https://job-boards.greenhouse.io/deepmind/jobs/7144793) |\n| [People Experience & Delivery Partner Bangalore, India](https://job-boards.greenhouse.io/deepmind/jobs/7090290) |\n| [People Partner, GenAI Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7112322) |\n| [Portfolio Lead, Google DeepMind Impact Accelerator Singapore](https://job-boards.greenhouse.io/deepmind/jobs/7126997) |\n| [Program Manager Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7092916) |\n| [Program Manager, Central Programs Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/6923549) |\n| [Program Manager, Product & Program Management (PPM) team London, UK](https://job-boards.greenhouse.io/deepmind/jobs/7126911) |\n| [Software Engineer, Org Tech London, UK](https://job-boards.greenhouse.io/deepmind/jobs/6897493) |\n| [Strategy and Operations Principal Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7019954) |\n| [Strategy & Operations Lead, Global Security Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7106388) |\n| [Strategy & Operations Senior Manager, Global Security London, UK](https://job-boards.greenhouse.io/deepmind/jobs/7106421) |\n| [Team Coordinator (12 Month Fixed Term Contract) London, UK](https://job-boards.greenhouse.io/deepmind/jobs/6980431) |\n| [Technical Program Manager Bangalore, India](https://job-boards.greenhouse.io/deepmind/jobs/7002564) |\n| [Technical Program Manager, Inception (12 Month Fixed Term Contract) London, UK](https://job-boards.greenhouse.io/deepmind/jobs/7019795) |\n| [Visual Designer New York City, New York, US](https://job-boards.greenhouse.io/deepmind/jobs/7064396) |\n\n### Foundational Research\n\n| Job |\n| --- |\n| [Research Engineer, Applied Robotics Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/6763910) |\n| [Research Engineer, AutoAI and Optimization Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7056350) |\n| [Research Engineer, Multimodal Generative AI, Google DeepMind New Singapore](https://job-boards.greenhouse.io/deepmind/jobs/7143783) |\n| [Research Engineer/Scientist, Training Algorithms Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7087309) |\n| [Research Engineer, Universal Knowledge Base Zurich, Switzerland](https://job-boards.greenhouse.io/deepmind/jobs/6668746) |\n| [Research Scientist, AnthroKrishi New Bangalore, India](https://job-boards.greenhouse.io/deepmind/jobs/7142337) |\n| [Research Scientist, Languages and Multimodality Bangalore, India](https://job-boards.greenhouse.io/deepmind/jobs/7089682) |\n| [Research Scientist, Languages and Multimodality Tokyo, Japan](https://job-boards.greenhouse.io/deepmind/jobs/7089543) |\n| [Research Scientist Lead, Singapore New Singapore](https://job-boards.greenhouse.io/deepmind/jobs/7130318) |\n| [Research Scientist, Multimodal Alignment, Safety, and Fairness Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7071457) |\n| [Research Scientist, Multimodal Generative AI, Google DeepMind New Singapore](https://job-boards.greenhouse.io/deepmind/jobs/7135034) |\n| [Research Scientist, Robotics Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7102795) |\n| [Software Engineer, Applied Robotics Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/6763890) |\n\n### GeminiApp\n\n| Job |\n| --- |\n| [Administrative Business Partner, Gemini New Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7155886) |\n| [AI Product Designer, Gemini Assistant Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/6674541) |\n| [Data Scientist, GeminiApp, Personalization Zurich, Switzerland](https://job-boards.greenhouse.io/deepmind/jobs/7065334) |\n| [Gemini App Product Manager, Mountain View Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/6848922) |\n| [GeminiApp, Senior Technical Program Manager New Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7142073) |\n| [Product Manager, Conversationality, Gemini App Mountain View, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7017685) |\n| [Product Manager, Gemini App for Devices Mountain View, California, US; San Francisco, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7094457) |\n| [Senior AI Product Designer, Gemini App Growth and Discovery Mountain View, California, US; New York City, New York, US; San Francisco, California, US](https://job-boards.greenhouse.io/deepmind/jobs/7120703) |\n\n*   1\n*   2\n\nPowered by\n\n[](https://www.greenhouse.io/privacy-policy)\n\nRead our[Privacy Policy](https://www.greenhouse.io/privacy-policy)\n"""
    
    
    # Get only the first proper name
    if proper_names:
        for name in proper_names:
            print(name)
            if name.lower() in web_content.lower():
                print(f"Found {name} in web content")
            else:
                print(f"Did not find {name} in web content")
    else:
        print("No proper names found")