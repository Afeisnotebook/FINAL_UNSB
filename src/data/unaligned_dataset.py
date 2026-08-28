import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class UnalignedDataset(BaseDataset):
    """Load official unpaired data or the frozen macro-marginal measure.

    ``macro_marginal`` samples A and B domain labels independently and
    uniformly, then samples an image uniformly inside each domain.  It is not
    DCUM: B is not conditioned on A's domain, no A/B mapping is stored, and
    no domain label is exposed to the generator or used at inference.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--macro_marginal',
            type=util.str2bool,
            nargs='?',
            const=True,
            default=False,
            help=(
                'training-only macro empirical measure: sample A and B domains '
                'independently and uniformly, with unpaired within-domain draws'
            ),
        )
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self._macro_enabled = (
            bool(getattr(opt, 'macro_marginal', False)) and bool(opt.isTrain)
        )
        self._A_by_domain = {}
        self._B_by_domain = {}
        self._macro_domains = []
        if self._macro_enabled:
            for path in self.A_paths:
                domain, _ = self._domain_and_stem(path)
                self._A_by_domain.setdefault(domain, []).append(path)
            for path in self.B_paths:
                domain, _ = self._domain_and_stem(path)
                self._B_by_domain.setdefault(domain, []).append(path)
            if set(self._A_by_domain) != set(self._B_by_domain):
                raise RuntimeError(
                    'macro marginal requires identical non-empty A/B domain sets'
                )
            self._macro_domains = sorted(self._A_by_domain)
            if not self._macro_domains:
                raise RuntimeError('macro marginal requires domain-prefixed files')

    @staticmethod
    def _domain_and_stem(path):
        """Read ``domain__stem`` from a materialized view without pairing it."""
        name = os.path.splitext(os.path.basename(path))[0]
        if '__' not in name:
            raise ValueError(
                'expected materialized filename domain__stem.ext: %s' % path
            )
        domain, stem = name.split('__', 1)
        return domain, stem

    def _sample_macro_pair(self):
        a_domain = self._macro_domains[
            random.randint(0, len(self._macro_domains) - 1)
        ]
        b_domain = self._macro_domains[
            random.randint(0, len(self._macro_domains) - 1)
        ]
        a_pool = self._A_by_domain[a_domain]
        b_pool = self._B_by_domain[b_domain]
        a_path = a_pool[random.randint(0, len(a_pool) - 1)]
        b_path = b_pool[random.randint(0, len(b_pool) - 1)]
        if a_domain == b_domain:
            _, a_stem = self._domain_and_stem(a_path)
            _, b_stem = self._domain_and_stem(b_path)
            if a_stem == b_stem:
                eligible = [
                    path for path in b_pool
                    if self._domain_and_stem(path)[1] != a_stem
                ]
                if not eligible:
                    raise RuntimeError(
                        'macro marginal has no different-stem B candidate in %s'
                        % a_domain
                    )
                b_path = eligible[random.randint(0, len(eligible) - 1)]
        return a_path, b_path

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self._macro_enabled:
            A_path, B_path = self._sample_macro_pair()
        else:
            A_path = self.A_paths[index % self.A_size]
            if self.opt.serial_batches:   # make sure index is within the range
                index_B = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
