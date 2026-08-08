"""Small, hand-curated, offline "Today in History" dataset.

No live API call and no ingestion pipeline -- entries are added by hand, the
same spirit as the grounding bootstrap notes (Council of Nicaea, etc.). This
spans Church history, general history, and philosophy/science -- not full
365-day coverage. By design: only one ecumenical council (Nicaea) and one
Didache entry are kept to avoid crowding the set with near-duplicates; Aquinas
appears only via his death date; and figures are represented by a monumental
achievement date rather than birth/death wherever a specific day is reliably
documented. A day with no entry means the homepage section is simply omitted
-- callers must not fabricate a placeholder for a missing date.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TodayInHistoryEntry:
    date: str  # "MM-DD"
    title: str
    description: str
    related_entity: str


TODAY_IN_HISTORY_ENTRIES: tuple[TodayInHistoryEntry, ...] = (
    TodayInHistoryEntry(
        date="01-01",
        title="Haiti declares independence",
        description=(
            "In 1804, Haiti became the first nation founded by a successful slave revolt, "
            "ending French colonial rule in Saint-Domingue."
        ),
        related_entity="Haitian Revolution",
    ),
    TodayInHistoryEntry(
        date="01-10",
        title="Julius Caesar crosses the Rubicon",
        description=(
            "In 49 BC, Caesar led his army across the Rubicon into Italy, an irreversible act "
            "of civil war against the Roman Senate that gave rise to the phrase \"crossing the "
            "Rubicon.\""
        ),
        related_entity="Julius Caesar",
    ),
    TodayInHistoryEntry(
        date="01-25",
        title="The traditional Feast of the Conversion of St. Paul",
        description=(
            "The Church marks Paul's turn from persecutor of Christians to apostle on the road "
            "to Damascus, an event recounted in Acts as a sudden reversal of conviction."
        ),
        related_entity="Paul the Apostle",
    ),
    TodayInHistoryEntry(
        date="01-26",
        title="The First Fleet arrives in Australia",
        description=(
            "In 1788, British ships carrying convicts and settlers landed at Sydney Cove, "
            "founding the first European colony on the continent."
        ),
        related_entity="Colonial Australia",
    ),
    TodayInHistoryEntry(
        date="01-27",
        title="Soviet forces liberate Auschwitz",
        description=(
            "In 1945, the Red Army entered the Auschwitz-Birkenau camp complex, ending its use "
            "as a site of mass killing during the Holocaust."
        ),
        related_entity="The Holocaust",
    ),
    TodayInHistoryEntry(
        date="02-11",
        title="Nelson Mandela is released from prison",
        description=(
            "In 1990, Mandela walked free after 27 years of imprisonment under apartheid, a "
            "turning point that led to South Africa's transition to democracy."
        ),
        related_entity="Nelson Mandela",
    ),
    TodayInHistoryEntry(
        date="02-20",
        title="John Glenn orbits the Earth",
        description=(
            "In 1962, Glenn became the first American to orbit the Earth, circling the planet "
            "three times aboard Friendship 7."
        ),
        related_entity="John Glenn",
    ),
    TodayInHistoryEntry(
        date="02-21",
        title="Marx and Engels publish The Communist Manifesto",
        description=(
            "Published in 1848, the pamphlet argued that history is driven by class struggle "
            "and called for the working class to overthrow capitalist society."
        ),
        related_entity="Karl Marx",
    ),
    TodayInHistoryEntry(
        date="03-07",
        title="Death of Thomas Aquinas",
        description=(
            "Thomas Aquinas died in 1274 at the Cistercian abbey of Fossanova while traveling "
            "to the Second Council of Lyon, leaving the Summa Theologiae unfinished."
        ),
        related_entity="Thomas Aquinas",
    ),
    TodayInHistoryEntry(
        date="03-09",
        title="Adam Smith publishes The Wealth of Nations",
        description=(
            "Published in 1776, Smith's account of markets, labor, and self-interest became "
            "the foundational text of modern economics."
        ),
        related_entity="Adam Smith",
    ),
    TodayInHistoryEntry(
        date="03-12",
        title="Gandhi begins the Salt March",
        description=(
            "In 1930, Gandhi set out on a 240-mile march to the sea to make salt in defiance of "
            "the British salt tax, a landmark act of nonviolent civil disobedience."
        ),
        related_entity="Mahatma Gandhi",
    ),
    TodayInHistoryEntry(
        date="03-25",
        title="Greece declares independence from the Ottoman Empire",
        description=(
            "In 1821, the Greek War of Independence began, eventually ending nearly four "
            "centuries of Ottoman rule over the Greek peninsula."
        ),
        related_entity="Greek War of Independence",
    ),
    TodayInHistoryEntry(
        date="03-26",
        title="Bangladesh declares independence",
        description=(
            "In 1971, Bangladesh declared independence from Pakistan, beginning a nine-month "
            "war that ended with the creation of a new nation."
        ),
        related_entity="Bangladesh Liberation War",
    ),
    TodayInHistoryEntry(
        date="04-06",
        title="The United States enters World War I",
        description=(
            "In 1917, Congress declared war on Germany, formally bringing American forces into "
            "a conflict that had already reshaped Europe for three years."
        ),
        related_entity="World War I",
    ),
    TodayInHistoryEntry(
        date="04-09",
        title="Lee surrenders at Appomattox Court House",
        description=(
            "In 1865, Confederate General Robert E. Lee surrendered to Union General Ulysses "
            "S. Grant, effectively ending the American Civil War."
        ),
        related_entity="American Civil War",
    ),
    TodayInHistoryEntry(
        date="04-12",
        title="The Civil War begins at Fort Sumter",
        description=(
            "In 1861, Confederate forces opened fire on the Union garrison at Fort Sumter, "
            "South Carolina, starting the American Civil War."
        ),
        related_entity="American Civil War",
    ),
    TodayInHistoryEntry(
        date="04-18",
        title="Paul Revere's ride, and the Battles of Lexington and Concord begin",
        description=(
            "On the night of April 18, 1775, Paul Revere rode to warn colonial militia of "
            "advancing British troops; fighting broke out the next morning, opening the "
            "American Revolutionary War."
        ),
        related_entity="American Revolutionary War",
    ),
    TodayInHistoryEntry(
        date="04-25",
        title="Watson and Crick publish the structure of DNA",
        description=(
            "In 1953, their paper in Nature proposed the double-helix structure of DNA, "
            "drawing on X-ray data from Rosalind Franklin and reshaping modern biology."
        ),
        related_entity="DNA",
    ),
    TodayInHistoryEntry(
        date="04-27",
        title="South Africa holds its first fully democratic election",
        description=(
            "In 1994, South Africans of all races voted for the first time, an election that "
            "brought Nelson Mandela to the presidency and formally ended apartheid."
        ),
        related_entity="Nelson Mandela",
    ),
    TodayInHistoryEntry(
        date="05-01",
        title="The Great Exhibition opens at the Crystal Palace",
        description=(
            "In 1851, London hosted the first world's fair, showcasing industrial and "
            "scientific achievements from around the globe under a purpose-built glass hall."
        ),
        related_entity="The Great Exhibition",
    ),
    TodayInHistoryEntry(
        date="05-06",
        title="The Eiffel Tower opens to the public",
        description=(
            "Completed for the 1889 World's Fair in Paris, the tower opened to visitors on "
            "this date, though it had been open to workers since March of that year."
        ),
        related_entity="Eiffel Tower",
    ),
    TodayInHistoryEntry(
        date="05-08",
        title="V-E Day: the war in Europe ends",
        description=(
            "In 1945, Germany's unconditional surrender took effect, ending nearly six years "
            "of war on the European continent."
        ),
        related_entity="World War II",
    ),
    TodayInHistoryEntry(
        date="05-10",
        title="The First Transcontinental Railroad is completed",
        description=(
            "In 1869, a golden spike was driven at Promontory Summit, Utah, joining the rail "
            "lines that connected the eastern and western United States."
        ),
        related_entity="Transcontinental Railroad",
    ),
    TodayInHistoryEntry(
        date="05-11",
        title="The Bryennios manuscript of the Didache is copied",
        description=(
            "The 1056 Constantinople manuscript that preserved the Didache's full text into "
            "the modern era, rediscovered by Philotheos Bryennios in 1873, was copied on this "
            "date, giving scholars their primary surviving witness to the text."
        ),
        related_entity="Didache",
    ),
    TodayInHistoryEntry(
        date="05-14",
        title="Israel declares independence",
        description=(
            "In 1948, David Ben-Gurion proclaimed the establishment of the State of Israel as "
            "the British Mandate for Palestine expired."
        ),
        related_entity="State of Israel",
    ),
    TodayInHistoryEntry(
        date="05-20",
        title="The Council of Nicaea convenes",
        description=(
            "In 325, bishops gathered at Nicaea at the summons of Constantine to settle the "
            "Arian controversy over the nature of Christ, producing the first form of the "
            "Nicene Creed."
        ),
        related_entity="Council of Nicaea",
    ),
    TodayInHistoryEntry(
        date="05-29",
        title="The fall of Constantinople",
        description=(
            "In 1453, Ottoman forces under Mehmed II captured Constantinople, ending the "
            "Byzantine Empire and marking a conventional boundary between the medieval and "
            "early modern periods."
        ),
        related_entity="Fall of Constantinople",
    ),
    TodayInHistoryEntry(
        date="06-04",
        title="Chinese authorities suppress protests at Tiananmen Square",
        description=(
            "In 1989, the Chinese government used military force to end weeks of pro-democracy "
            "demonstrations centered on Tiananmen Square in Beijing."
        ),
        related_entity="Tiananmen Square protests",
    ),
    TodayInHistoryEntry(
        date="06-05",
        title="The Marshall Plan is announced",
        description=(
            "In 1947, U.S. Secretary of State George Marshall proposed a program of American "
            "economic aid to help rebuild Western Europe after World War II."
        ),
        related_entity="Marshall Plan",
    ),
    TodayInHistoryEntry(
        date="06-06",
        title="D-Day: Allied forces land in Normandy",
        description=(
            "In 1944, Allied troops landed on the beaches of Normandy, opening a new front in "
            "Western Europe against Nazi Germany."
        ),
        related_entity="D-Day",
    ),
    TodayInHistoryEntry(
        date="06-08",
        title="Descartes publishes Discourse on the Method",
        description=(
            "Published in 1637, Descartes's work introduced radical doubt as a method for "
            "arriving at certain knowledge, giving modern philosophy its \"I think, therefore "
            "I am.\""
        ),
        related_entity="René Descartes",
    ),
    TodayInHistoryEntry(
        date="06-15",
        title="Magna Carta is sealed",
        description=(
            "In 1215, King John of England sealed the Magna Carta at Runnymede, establishing "
            "the principle that even the monarch was bound by law."
        ),
        related_entity="Magna Carta",
    ),
    TodayInHistoryEntry(
        date="06-18",
        title="The Battle of Waterloo",
        description=(
            "In 1815, a coalition led by Britain and Prussia defeated Napoleon Bonaparte near "
            "Waterloo, ending his final bid to rule France and Europe."
        ),
        related_entity="Napoleon Bonaparte",
    ),
    TodayInHistoryEntry(
        date="06-19",
        title="The Council of Nicaea concludes",
        description=(
            "The council's business closed in 325 with the bishops' formal subscriptions to "
            "the creed condemning Arius's teaching that the Son was a created being."
        ),
        related_entity="Council of Nicaea",
    ),
    TodayInHistoryEntry(
        date="06-22",
        title="Galileo recants before the Roman Inquisition",
        description=(
            "In 1633, Galileo was compelled to renounce his support for a sun-centered solar "
            "system, a confrontation that became a lasting symbol of tension between science "
            "and authority."
        ),
        related_entity="Galileo Galilei",
    ),
    TodayInHistoryEntry(
        date="06-28",
        title="The Treaty of Versailles is signed",
        description=(
            "In 1919, the treaty formally ending World War I imposed harsh terms on Germany, "
            "consequences that later fed into the outbreak of World War II."
        ),
        related_entity="Treaty of Versailles",
    ),
    TodayInHistoryEntry(
        date="06-30",
        title="Einstein submits his special relativity paper",
        description=(
            "In 1905, Einstein submitted \"On the Electrodynamics of Moving Bodies,\" "
            "introducing special relativity and overturning the classical understanding of "
            "space and time."
        ),
        related_entity="Albert Einstein",
    ),
    TodayInHistoryEntry(
        date="07-01",
        title="Canada becomes a self-governing dominion",
        description=(
            "In 1867, the British North America Act took effect, uniting several colonies "
            "into the Dominion of Canada."
        ),
        related_entity="Canada",
    ),
    TodayInHistoryEntry(
        date="07-04",
        title="The U.S. Declaration of Independence is adopted",
        description=(
            "In 1776, the Second Continental Congress adopted the Declaration of Independence, "
            "whose language on natural rights drew heavily on Enlightenment political "
            "philosophy."
        ),
        related_entity="Declaration of Independence",
    ),
    TodayInHistoryEntry(
        date="07-05",
        title="Newton's Principia Mathematica is published",
        description=(
            "In 1687, Newton's Philosophiæ Naturalis Principia Mathematica laid out the laws "
            "of motion and universal gravitation, unifying terrestrial and celestial physics."
        ),
        related_entity="Isaac Newton",
    ),
    TodayInHistoryEntry(
        date="07-14",
        title="The storming of the Bastille",
        description=(
            "In 1789, Parisians stormed the Bastille fortress, a flashpoint that opened the "
            "French Revolution and its upheaval of the old political and social order."
        ),
        related_entity="French Revolution",
    ),
    TodayInHistoryEntry(
        date="07-16",
        title="The Great Schism between Rome and Constantinople",
        description=(
            "In 1054, mutual excommunications between papal legates and the Patriarch of "
            "Constantinople marked the formal, lasting split between the Western and Eastern "
            "churches."
        ),
        related_entity="East-West Schism",
    ),
    TodayInHistoryEntry(
        date="08-04",
        title="France's National Assembly abolishes feudal privileges",
        description=(
            "In the August Decrees of 1789, the Assembly swept away feudal dues, tithes, and "
            "noble privileges in a single overnight session during the French Revolution."
        ),
        related_entity="French Revolution",
    ),
    TodayInHistoryEntry(
        date="08-06",
        title="The atomic bombing of Hiroshima",
        description=(
            "In 1945, the United States dropped an atomic bomb on Hiroshima, Japan, the first "
            "use of a nuclear weapon in warfare."
        ),
        related_entity="World War II",
    ),
    TodayInHistoryEntry(
        date="08-15",
        title="India gains independence from Britain",
        description=(
            "In 1947, India became independent after nearly two centuries of British rule, "
            "coinciding with the partition that created Pakistan."
        ),
        related_entity="Indian independence movement",
    ),
    TodayInHistoryEntry(
        date="08-18",
        title="The 19th Amendment granting women's suffrage is ratified",
        description=(
            "In 1920, the amendment prohibiting the denial of voting rights on account of sex "
            "was ratified, extending the franchise to women across the United States."
        ),
        related_entity="Women's suffrage",
    ),
    TodayInHistoryEntry(
        date="08-24",
        title="The traditional date for the eruption of Vesuvius",
        description=(
            "Mount Vesuvius's eruption in 79 AD buried Pompeii and Herculaneum in ash, "
            "preserving a detailed record of daily life in the Roman world; some scholarship "
            "suggests an autumn date instead."
        ),
        related_entity="Pompeii",
    ),
    TodayInHistoryEntry(
        date="08-26",
        title="France adopts the Declaration of the Rights of Man",
        description=(
            "In 1789, the National Constituent Assembly adopted a statement of universal "
            "rights that became a foundational document of the French Revolution."
        ),
        related_entity="French Revolution",
    ),
    TodayInHistoryEntry(
        date="08-28",
        title="Martin Luther King Jr. delivers \"I Have a Dream\"",
        description=(
            "In 1963, King addressed the March on Washington for Jobs and Freedom, calling for "
            "racial equality in one of the most quoted speeches in American history."
        ),
        related_entity="Martin Luther King Jr.",
    ),
    TodayInHistoryEntry(
        date="09-02",
        title="V-J Day: Japan formally surrenders",
        description=(
            "In 1945, Japanese officials signed the instrument of surrender aboard the USS "
            "Missouri, formally ending World War II."
        ),
        related_entity="World War II",
    ),
    TodayInHistoryEntry(
        date="09-04",
        title="The traditional date for the fall of the Western Roman Empire",
        description=(
            "In 476, the Germanic general Odoacer deposed the last Western Roman emperor, "
            "Romulus Augustulus, an event later historians took as marking the empire's end."
        ),
        related_entity="Fall of the Roman Empire",
    ),
    TodayInHistoryEntry(
        date="09-11",
        title="The September 11 terrorist attacks",
        description=(
            "In 2001, coordinated attacks on the World Trade Center and the Pentagon killed "
            "nearly 3,000 people and reshaped American foreign and domestic policy for "
            "decades."
        ),
        related_entity="September 11 attacks",
    ),
    TodayInHistoryEntry(
        date="09-14",
        title="Marx publishes Das Kapital, Volume I",
        description=(
            "Published in 1867, Marx's critique of political economy analyzed capitalism's "
            "internal contradictions and became the theoretical core of later socialist "
            "movements."
        ),
        related_entity="Karl Marx",
    ),
    TodayInHistoryEntry(
        date="09-17",
        title="The U.S. Constitution is signed",
        description=(
            "In 1787, delegates to the Constitutional Convention in Philadelphia signed the "
            "document establishing the federal framework of American government."
        ),
        related_entity="U.S. Constitution",
    ),
    TodayInHistoryEntry(
        date="09-28",
        title="Fleming's discovery of penicillin, traditionally dated",
        description=(
            "Alexander Fleming is traditionally credited with noticing, around this date in "
            "1928, that a mold contaminating one of his cultures was killing surrounding "
            "bacteria -- the discovery that led to the first antibiotic."
        ),
        related_entity="Alexander Fleming",
    ),
    TodayInHistoryEntry(
        date="10-04",
        title="The Soviet Union launches Sputnik 1",
        description=(
            "In 1957, Sputnik 1 became the first artificial satellite to orbit the Earth, "
            "opening the Space Age and the Cold War space race."
        ),
        related_entity="Sputnik 1",
    ),
    TodayInHistoryEntry(
        date="10-12",
        title="Columbus reaches the Americas",
        description=(
            "In 1492, Christopher Columbus's expedition sighted land in the Bahamas, beginning "
            "sustained European contact with the Americas."
        ),
        related_entity="Christopher Columbus",
    ),
    TodayInHistoryEntry(
        date="10-14",
        title="The Battle of Hastings",
        description=(
            "In 1066, William of Normandy defeated King Harold II at Hastings, opening the "
            "Norman Conquest of England."
        ),
        related_entity="Battle of Hastings",
    ),
    TodayInHistoryEntry(
        date="10-24",
        title="The United Nations Charter takes effect",
        description=(
            "In 1945, the UN Charter entered into force after ratification by its founding "
            "members, formally establishing the United Nations."
        ),
        related_entity="United Nations",
    ),
    TodayInHistoryEntry(
        date="10-29",
        title="The Wall Street Crash begins",
        description=(
            "Beginning on this day in 1929, a sharp collapse in U.S. stock prices set off the "
            "chain of events that led into the Great Depression."
        ),
        related_entity="Great Depression",
    ),
    TodayInHistoryEntry(
        date="10-31",
        title="Martin Luther posts the Ninety-Five Theses",
        description=(
            "In 1517, Luther's disputation against the sale of indulgences, posted in "
            "Wittenberg, is traditionally dated to this day and is widely marked as the start "
            "of the Protestant Reformation."
        ),
        related_entity="Martin Luther",
    ),
    TodayInHistoryEntry(
        date="11-07",
        title="The October Revolution begins in Petrograd",
        description=(
            "In 1917 (by the Gregorian calendar), Bolshevik forces seized power in Petrograd, "
            "leading to the founding of the Soviet state."
        ),
        related_entity="Russian Revolution",
    ),
    TodayInHistoryEntry(
        date="11-09",
        title="The fall of the Berlin Wall",
        description=(
            "In 1989, East German authorities opened the border crossings, and crowds began "
            "tearing down the Berlin Wall, a symbolic end to the Cold War division of Europe."
        ),
        related_entity="Berlin Wall",
    ),
    TodayInHistoryEntry(
        date="11-19",
        title="Lincoln delivers the Gettysburg Address",
        description=(
            "In 1863, at the dedication of a military cemetery, Lincoln reframed the Civil War "
            "in a brief speech invoking the principle that all men are created equal."
        ),
        related_entity="Abraham Lincoln",
    ),
    TodayInHistoryEntry(
        date="11-20",
        title="The Nuremberg trials begin",
        description=(
            "In 1945, an international tribunal opened proceedings against senior Nazi "
            "officials, establishing precedents for prosecuting war crimes and crimes against "
            "humanity."
        ),
        related_entity="Nuremberg trials",
    ),
    TodayInHistoryEntry(
        date="11-22",
        title="President Kennedy is assassinated in Dallas",
        description=(
            "In 1963, President John F. Kennedy was shot and killed while riding in a "
            "motorcade in Dallas, Texas, a moment that shaped a generation's view of American "
            "public life."
        ),
        related_entity="John F. Kennedy",
    ),
    TodayInHistoryEntry(
        date="11-24",
        title="Darwin publishes On the Origin of Species",
        description=(
            "Published in 1859, Darwin's book set out the theory of evolution by natural "
            "selection, reshaping biology and the broader understanding of humanity's place in "
            "nature."
        ),
        related_entity="Charles Darwin",
    ),
    TodayInHistoryEntry(
        date="12-01",
        title="Rosa Parks refuses to give up her seat",
        description=(
            "In 1955, Parks's refusal to give up her bus seat to a white passenger in "
            "Montgomery, Alabama, sparked a boycott that became a catalyst of the American "
            "civil rights movement."
        ),
        related_entity="Rosa Parks",
    ),
    TodayInHistoryEntry(
        date="12-06",
        title="The 13th Amendment abolishing slavery is ratified",
        description=(
            "In 1865, ratification of the 13th Amendment formally abolished slavery throughout "
            "the United States."
        ),
        related_entity="13th Amendment",
    ),
    TodayInHistoryEntry(
        date="12-07",
        title="The attack on Pearl Harbor",
        description=(
            "In 1941, Japan launched a surprise attack on the U.S. naval base at Pearl Harbor, "
            "bringing the United States into World War II."
        ),
        related_entity="Pearl Harbor",
    ),
    TodayInHistoryEntry(
        date="12-10",
        title="The UN adopts the Universal Declaration of Human Rights",
        description=(
            "In 1948, the United Nations General Assembly adopted a declaration setting out "
            "rights held to belong to every person, drafted in the aftermath of World War II."
        ),
        related_entity="Universal Declaration of Human Rights",
    ),
    TodayInHistoryEntry(
        date="12-15",
        title="The U.S. Bill of Rights is ratified",
        description=(
            "In 1791, the first ten amendments to the U.S. Constitution were ratified, "
            "guaranteeing rights including speech, religion, and due process."
        ),
        related_entity="Bill of Rights",
    ),
    TodayInHistoryEntry(
        date="12-16",
        title="The Boston Tea Party",
        description=(
            "In 1773, colonists boarded British ships in Boston Harbor and destroyed a "
            "shipment of tea in protest of taxation without representation, escalating "
            "tensions toward the American Revolution."
        ),
        related_entity="American Revolutionary War",
    ),
    TodayInHistoryEntry(
        date="12-17",
        title="The Wright Brothers' first powered flight",
        description=(
            "In 1903, Orville and Wilbur Wright achieved the first sustained, controlled "
            "flight of a powered aircraft at Kitty Hawk, North Carolina."
        ),
        related_entity="Wright Brothers",
    ),
    TodayInHistoryEntry(
        date="12-25",
        title="Charlemagne is crowned Holy Roman Emperor",
        description=(
            "On Christmas Day in 800, Pope Leo III crowned Charlemagne emperor in Rome, an act "
            "later seen as founding the Holy Roman Empire and reviving the idea of an imperial "
            "office in the West."
        ),
        related_entity="Charlemagne",
    ),
)

_BY_DATE: dict[str, TodayInHistoryEntry] = {entry.date: entry for entry in TODAY_IN_HISTORY_ENTRIES}


def entry_for_date(month_day: str) -> TodayInHistoryEntry | None:
    return _BY_DATE.get(month_day)
