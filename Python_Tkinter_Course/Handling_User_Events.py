# Handling_User_Events
""" Event handling is the process of executing specific functions when
    certain inputs are received from the mouse, keyboard and other sources.

    Some of the events:
        1. ButtonPress - mouse clicks
        2. key - key strokes
        3. Leave - widget interaction
        4. Motion - mouse movements
        5. Configure - like window resizing

    For each event handler code executes:
        1. handle_buttonpress()
        2. handle_key()
        3. handdle_leave()
        4. handle_motion()
        5. handle_configure()

    When you run the root.mainloo(), {event loop} is initiated.
    This waits for the event to occur and as soon as the event occurs
    the appropriate handler code is called. When the handler is exceuted
    the program comes back to the Event Loop and waits for the next event
    to occur.

    ** Event Loop is not able to handle multiple events at the sametime.

    There are two primary ways to configure Event Handlers:
        1. Commands callbacks - interactive widgets, which can
                take commands.
            example: An example of a widget with command
                property is button, as it is unlikley for
                one to create a button which does nothing.
        2. Event bindings - for widgets which dont have command
                properties/callback.
            example: labels widgets can use bindings.

** If you want it to execute some handler code when the user puts
    the mouse over the label, you could bind it with the enter event
    which is triggered when the mouse enters the region over the label.
    There are still a variety of events that you can bind to which are
    related to keyboard and mouse actions. We'll cover those in detail
    in the later event binding section of the course.
