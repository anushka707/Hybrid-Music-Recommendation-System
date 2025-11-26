import pandas as pd
from .config import DATA_RAW


def load_all_data():
    tracks = pd.read_csv(f"{DATA_RAW}/tracks.dat", sep="\t")
    artists = pd.read_csv(f"{DATA_RAW}/artists.dat", sep="\t")
    tags = pd.read_csv(f"{DATA_RAW}/tags.dat", sep="\t")
    user_tagged = pd.read_csv(f"{DATA_RAW}/user_taggedtracks.dat", sep="\t")
    return tracks, artists, tags, user_tagged


def merge_tracks_artists_tags():
    tracks, artists, tags, user_tagged = load_all_data()

    # add artist name
    tracks = tracks.merge(
        artists[["id", "name"]],
        left_on="artistID", right_on="id",
        how="left"
    ).rename(columns={"name": "artist_name"})

    # aggregate tags
    tags_dict = dict(zip(tags["id"], tags["tagValue"]))

    user_tagged["tagValue"] = user_tagged["tagID"].map(tags_dict)

    tag_groups = user_tagged.groupby("trackID")["tagValue"].apply(
        lambda vals: " ".join([v for v in vals if pd.notna(v)])
    ).reset_index()

    tracks = tracks.merge(
        tag_groups,
        left_on="id",
        right_on="trackID",
        how="left"
    )

    tracks["tagValue"] = tracks["tagValue"].fillna("")

    # final text description
    tracks["text"] = (
        tracks["title"].astype(str) + " " +
        tracks["artist_name"].astype(str) + " " +
        tracks["tagValue"].astype(str)
    )

    return tracks[["id", "artistID", "title", "artist_name", "text"]]
