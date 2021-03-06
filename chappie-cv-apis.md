## Chappie Computer Vison APIs

HTTP/1.1

Host URL: `cv.chappiebot.com`

Header:
    X-Api-Key: ...


### Labelling an image

POST https://cv.chappiebot.com/v2/labelling

Params:
```
    file: attacted in post request with format jpg/png, require minimum size 200x100 pixel and maximum 10 files at a time.
    request_id: optional
```

Response:
JSON format
```python
{
    request_id: <request_id>,
    predictions: [
                {
                    filename: <filename>,
                    objects: [
                        CAR, PEOPLE, DOG, ...
                    ],
                    describe: {
                        type: <'exterior', 'interior', 'engine'>
                        view: <'e_door_handle', 'e_front_view', 'e_grille', 'e_headlight', 'e_left_front_view', 'e_left_rear_view',
                            'e_left_view', 'e_rear_light', 'e_rear_view', 'e_right_front_view', 'e_right_rear_view', 'e_right_view',
                            'e_side_mirror', 'e_sun_roof', 'e_top_view', 'e_wheel',
                            'i_rear_view_mirror', 'i_driving_seat_side_view', 'i_full_interior_view', 'i_gear_stick', 'i_media_screen',
                            'i_odometer', 'i_rear_seat_front_view', 'i_rear_seat_side_view', 'i_steering_wheel',
                            'i_steering_wheel_dash_board', 'i_sun_roof_view', 'i_trunk', 'i_button', 'i_door_handle'>
                        marks: [
                            (CAR, (x1,y1,x2,y2)),
                            (PEOPLE, (x1,y1,x2,y2)),
                            ...
                        ]
                    }
                },
                ...
            ]
}
```

Example:
```json
    {
        "request_id": 123232323223,
        "predictions": [
            {
                "filename": "image1.jpg",
                "objects": ["CAR"],
                "describe": {
                    "type": "exterior",
                    "view": "e_rear_view",
                    "marks": [
                        ["CAR", [45,37,218,189]]
                    ]
                }
            }
        ]
    }
```
