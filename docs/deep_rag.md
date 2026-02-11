# Abstract Deep RAG

The deep_rag agent network is a agent network that, given a list of text files
encompassing a larger corpus of text, will generate a set of _other_ agent networks which
allow you to make queries on the corpus as a whole with more detailed knowledge than
a simple RAG approach can provide.

# Before you run

The deep_rag agent network uses neuro-san Reservations, so you will need to have
an environment variable set in order to allow those to be used:

```bash
export AGENT_TEMPORARY_NETWORK_UPDATE_PERIOD_SECONDS=5
```

## Description

The deep_rag network uses the neuro-san concept of Reservations to create AAOSA-oriented
temporary networks that pertain to the files listed in the user's corpus.

Since it helps to have knowledge about how the files themselves are organized, the deep_rag
appreciates the following information in your interaction with it:

* The name of the corpus you are dealing with
* Where the files are located on the local file system
* How the files are organized and which components of file names/paths are important for
  grouping specialist agents together.
* Any instructions about grouping contraints you would like to give.
* The list of files themselves.

The deep_rag agent network will specifically create a 3-tiered agent structure across
several specific agent networks, where:

* Any single grouping of content gets its own agent subnetwork which contains two of the tiers:
   * Each leaf node pertains to an expert on a particular file of the corpus.
   * The front man of each network becomes an AAOSA aggregator for queries to the experts
* Another network is created to be the main entry point for queries against the entire corpus.
  This guy's frontman knows to call out to all the other subnetworks for groups described above.


## Example conversation

First you interact with the deep_rag agent to build your temporary corpus network:

```text
HUMAN:

I will give you files pertaining to a play called "Hamlet".

The files are in the directory: /<your_local_path>/deep_rag_docs/hamlet

File name structure:
* Any one file name will have a numbering system as a prefix:
    1 The 1st number of the prefix always represents the act number of the play.
      Anything part of Act 0 includes preparatory information about the play, but is not the play itself.
    2 The 2nd number of the prefix always represents the scene number of the act
    3 The 3rd number (if present) represents a change of focus within the same scene to a different group of characters.

* After the numeric prefix of a single file name, there is a very short description of what transpires
  in that scene or part of a scene.

* The ".txt" suffix that each file ends with has no bearing as to the content
  within the file other than its format and should not be considered for organizational purposes.

Grouping Constraints:
* Never mix files from different Acts in the same group.
* It's OK if a single Act needs more than one group, but only do this for numeric constraints
  or more consistent logical groupings.  If Act splitting is necessary, try to split the Act
  up such that each groups has the same number of files.

The list of files to use in order are:

1.1-Elsinore.txt
1.2-Within-the-castle.txt
1.3-In-the-house-of-Polonius.txt
1.4-The-platform.txt
1.5-Another-part-of-the-platform.txt
2.1-Room-of-Polonius.txt
2.2.1-A-room-in-the-castle.txt
2.2.2-Hamlet-and-pals.txt
3.1-A-room-in-the-castle.txt
3.2-A-hall-in-the-castle.txt
3.3-A-room-in-the-castle.txt
3.4-The-queens-closet.txt
4.1-A-room-in-the-castle.txt
4.2-Another-room-in-the-castle.txt
4.3-Another-room-in-the-castle.txt
4.4-A-plain-in-Denmark.txt
4.5-A-room-in-Elsinore-castle.txt
4.6-Another-room-in-the-castle.txt
4.7-Another-room-in-the-castle.txt
5.1-A-churchyard.txt
5.2-A-hall-in-the-castle.txt
```

The deep_rag agent will respond with two bits of information returned in its sly_data:
* aa_grouping_json gives you information as to how it decided to group all of your files
  as well as how it decided to translate the file names to agent names that represent the files.
* agent_reservations gives you a list of temporary agent network dictionaries that were created
  as a result of your deep_rag query.  Most of these will be specialist networks, and there
  is nothing wrong with you querying those directly.  The last one listed, however is the
  entry point for the whole corpus. (To be continued on the other side of the example response).

```text
AI:
The network has been successfully created. You can access your deep rag network using the agent named `hamlet_scene_division-5f1ff4e6-c256-403c-87fa-43667ea1ed6f`. Please note that this access is available for the next 3600 seconds.
Returned sly_data is: {
    "aa_grouping_json": {
        "description": "The content is grouped by Act, ensuring each group contains scenes from the same Act. If an Act has more than 7 files, it is split into multiple groups with balanced content distribution.",
        "groups": [
            {
                "description": "Scenes from Act 1, which consists of 5 unique locations.",
                "files": {
                    "1.1-Elsinore.txt": "1_1_elsinore",
                    "1.2-Within-the-castle.txt": "1_2_within_the_castle",
                    "1.3-In-the-house-of-Polonius.txt": "1_3_in_the_house_of_polonius",
                    "1.4-The-platform.txt": "1_4_the_platform",
                    "1.5-Another-part-of-the-platform.txt": "1_5_another_part_of_the_platform"
                },
                "name": "act_1_scenes"
            },
            {
                "description": "Scenes from Act 2, which includes the room of Polonius and a room in the castle.",
                "files": {
                    "2.1-Room-of-Polonius.txt": "2_1_room_of_polonius",
                    "2.2.1-A-room-in-the-castle.txt": "2_2_1_a_room_in_the_castle",
                    "2.2.2-Hamlet-and-pals.txt": "2_2_2_hamlet_and_pals"
                },
                "name": "act_2_scenes"
            },
            {
                "description": "Scenes from Act 3, featuring multiple locations within the castle and the queen's closet.",
                "files": {
                    "3.1-A-room-in-the-castle.txt": "3_1_a_room_in_the_castle",
                    "3.2-A-hall-in-the-castle.txt": "3_2_a_hall_in_the_castle",
                    "3.3-A-room-in-the-castle.txt": "3_3_a_room_in_the_castle",
                    "3.4-The-queens-closet.txt": "3_4_the_queens_closet"
                },
                "name": "act_3_scenes"
            },
            {
                "description": "First group of scenes from Act 4, focusing on various rooms in the castle and the plain in Denmark.",
                "files": {
                    "4.1-A-room-in-the-castle.txt": "4_1_a_room_in_the_castle",
                    "4.2-Another-room-in-the-castle.txt": "4_2_another_room_in_the_castle",
                    "4.3-Another-room-in-the-castle.txt": "4_3_another_room_in_the_castle",
                    "4.4-A-plain-in-Denmark.txt": "4_4_a_plain_in_denmark"
                },
                "name": "act_4_group_1"
            },
            {
                "description": "Second group of scenes from Act 4, mostly centered in Elsinore castle.",
                "files": {
                    "4.5-A-room-in-Elsinore-castle.txt": "4_5_a_room_in_elsinore_castle",
                    "4.6-Another-room-in-the-castle.txt": "4_6_another_room_in_the_castle",
                    "4.7-Another-room-in-the-castle.txt": "4_7_another_room_in_the_castle"
                },
                "name": "act_4_group_2"
            },
            {
                "description": "Scenes from Act 5, covering the churchyard and a hall in the castle.",
                "files": {
                    "5.1-A-churchyard.txt": "5_1_a_churchyard",
                    "5.2-A-hall-in-the-castle.txt": "5_2_a_hall_in_the_castle"
                },
                "name": "act_5_scenes"
            }
        ],
        "name": "hamlet_scene_division"
    },
    "agent_reservations": [
        {
            "expiration_time_in_seconds": 1761872603.6403883,
            "lifetime_in_seconds": 3600,
            "reservation_id": "act_1_scenes-43c37ca2-6083-4f0e-8849-7c3ba16cc829"
        },
        {
            "expiration_time_in_seconds": 1761872603.6403883,
            "lifetime_in_seconds": 3600,
            "reservation_id": "act_2_scenes-4e3f471f-dc2f-461c-9898-0cf399fbc16d"
        },
        {
            "expiration_time_in_seconds": 1761872603.6403883,
            "lifetime_in_seconds": 3600,
            "reservation_id": "act_3_scenes-31834d6e-4ea9-46f3-856d-cd02e0df0055"
        },
        {
            "expiration_time_in_seconds": 1761872603.6403883,
            "lifetime_in_seconds": 3600,
            "reservation_id": "act_4_group_1-6e30274c-8dc3-449e-bb6e-a36d4445f11c"
        },
        {
            "expiration_time_in_seconds": 1761872603.6403883,
            "lifetime_in_seconds": 3600,
            "reservation_id": "act_4_group_2-1828cf2e-cfd0-4ee9-ae68-f8ce5bbc9809"
        },
        {
            "expiration_time_in_seconds": 1761872603.6403883,
            "lifetime_in_seconds": 3600,
            "reservation_id": "act_5_scenes-4977c51a-b2d4-47ac-86b2-9dfd8ae15f53"
        },
        {
            "expiration_time_in_seconds": 1761872603.6403883,
            "lifetime_in_seconds": 3600,
            "reservation_id": "hamlet_scene_division-5f1ff4e6-c256-403c-87fa-43667ea1ed6f"
        }
    ]
}
```

It is this last agent listing's reservation_id that becomes who you talk to next to query the corpus:

For this example we talk to the agent named 'hamlet_scene_division-5f1ff4e6-c256-403c-87fa-43667ea1ed6f',
but since these big long strings have UUIDs in them, your mileage will vary on that agent name:

```text
HUMAN:
Which scenes contain some form of the word "rot"?
```

```text
AI:
The scenes containing some form of the word "rot" in the play "Hamlet" are as follows:

- **Act 1, Scene IV**: Contains the phrase "Something is rotten in the state of Denmark."
- **Act 5, Scene I**: Contains the phrase "How long will a man lie i' the earth ere he rot?"

These are the only scenes across all acts where a form of the word "rot" appears.
```
... which is pretty good, even though it missed one reference in the same scene.


```text
HUMAN:
Which character tends to speak with the most semicolons?
```

```text
AI:
The inquiry about which character tends to speak with the most semicolons throughout the play "Hamlet" has been addressed by experts from different acts, and the findings are as follows:

1. **Act 1**: Different characters across scenes use semicolons frequently, including Horatio, King Claudius, Polonius, Hamlet, and the Ghost. Each scene features a different character, suggesting no single standout across the act.

2. **Act 2**: Polonius frequently uses semicolons in his detailed instructions and explanations, while Hamlet also uses them to transition between thoughts.

3. **Act 3**: Hamlet is noted for frequently using semicolons in his speeches.

4. **Act 4**: King Claudius and Hamlet are notable users of semicolons in their respective dialogues throughout different scenes.

5. **Act 5**: Hamlet frequently uses semicolons in both the churchyard scene and the hall in the castle, where he connects related thoughts and ideas.

Overall, Hamlet emerges as a strong candidate for using semicolons frequently, especially noted in Acts 3, 4, and 5. However, for a definitive answer, a comprehensive analysis across the entire play would be necessary to determine if he uses semicolons more than any other character.
```
This is something where even though a single LLM's might be trained on the elements of the corpus,
it's not going to have the actual corpus memorized, and these kinds of things is where deep_rag networks shine.

# Limitations / Future work

* The deep_rag network seems to be overly cautious about answering questions that compare or relate
  one section of the corpus to another, preferring to wuss out due to its prompting about not making things
  up outside of what the experts tell it.

  In the future we'd like to experiment with creating a top-level analysis agent which is free from
  those constraints.

* See issues filed in this repo with the Deep RAG heading for more future improvements.
