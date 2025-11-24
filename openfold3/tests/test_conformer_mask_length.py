# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for conformer mask length matching after molecule modifications."""

import unittest
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
    get_processed_reference_conformer,
)
from openfold3.core.data.primitives.structure.labels import uniquify_ids


class TestConformerMaskLength(unittest.TestCase):
    """Test that in_crop_mask length matches molecule atom count after modifications."""

    def _create_test_molecule_with_hydrogens(self, smiles=None):
        """Create a test molecule with hydrogens.
        
        Args:
            smiles: SMILES string for the molecule. If None, uses ethanol (CCO).
        
        Returns:
            RDKit Mol object with hydrogens added and a conformer.
        """
        if smiles is None:
            smiles = "CCO"  # Ethanol (default)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)  # Add hydrogens
        Chem.SanitizeMol(mol)
        
        # Add atom names
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetProp("annot_atom_name", f"{atom.GetSymbol()}{i+1}")
        
        # Create a conformer
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
        mol.AddConformer(conf)
        
        return mol

    def _create_test_atom_array(self, mol):
        """Create a test AtomArray from molecule."""
        num_atoms = mol.GetNumAtoms()
        atom_array = AtomArray(num_atoms)
        atom_array.element = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
        atom_array.coord = np.array([
            list(mol.GetConformer().GetAtomPosition(i)) for i in range(num_atoms)
        ])
        atom_array.chain_id = np.array(["A"] * num_atoms)
        atom_array.res_id = np.array([1] * num_atoms)
        atom_array.res_name = np.array(["LIG"] * num_atoms)
        atom_array.atom_name = np.array([
            atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()
        ])
        atom_array.component_id = np.array([1] * num_atoms)
        
        # Create crop mask (all atoms in crop for this test)
        atom_array.crop_mask = np.ones(num_atoms, dtype=bool)
        
        # Add atom_name_unique annotation (uniquified)
        atom_array.atom_name_unique = np.array(uniquify_ids(atom_array.atom_name.tolist()))
        
        return atom_array

    def test_mask_length_after_conformer_generation_with_hydrogen_removal(self):
        """Test that in_crop_mask length matches atoms after hydrogen removal."""
        mol = self._create_test_molecule_with_hydrogens()
        atom_array = self._create_test_atom_array(mol)
        
        # Get initial atom count (with hydrogens)
        initial_atom_count = mol.GetNumAtoms()
        
        # Create processed reference conformer with default strategy (removes Hs)
        processed_mol = get_processed_reference_conformer(
            mol=mol,
            mol_atom_array=atom_array,
            preferred_confgen_strategy="default",
            set_fallback_to_nan=False,
        )
        
        # Check that mask length matches current molecule atom count
        current_atom_count = processed_mol.mol.GetNumAtoms()
        mask_length = len(processed_mol.in_crop_mask)
        
        self.assertEqual(
            mask_length,
            current_atom_count,
            f"in_crop_mask length ({mask_length}) does not match molecule atom count "
            f"({current_atom_count}). Initial count was {initial_atom_count}."
        )
        
        # Verify we can iterate without error
        for atom, mask in zip(processed_mol.mol.GetAtoms(), processed_mol.in_crop_mask, strict=True):
            self.assertIsNotNone(atom)
            self.assertIsInstance(mask, (bool, np.bool_))

    def test_mask_length_with_random_init_strategy(self):
        """Test that mask length is correct with random_init strategy."""
        mol = self._create_test_molecule_with_hydrogens()
        atom_array = self._create_test_atom_array(mol)
        
        processed_mol = get_processed_reference_conformer(
            mol=mol,
            mol_atom_array=atom_array,
            preferred_confgen_strategy="random_init",
            set_fallback_to_nan=False,
        )
        
        current_atom_count = processed_mol.mol.GetNumAtoms()
        mask_length = len(processed_mol.in_crop_mask)
        
        self.assertEqual(
            mask_length,
            current_atom_count,
            f"in_crop_mask length ({mask_length}) does not match molecule atom count "
            f"({current_atom_count}) with random_init strategy."
        )

    def test_mask_length_with_fallback_strategy(self):
        """Test that mask length is correct when using fallback conformer."""
        mol = self._create_test_molecule_with_hydrogens()
        atom_array = self._create_test_atom_array(mol)
        
        processed_mol = get_processed_reference_conformer(
            mol=mol,
            mol_atom_array=atom_array,
            preferred_confgen_strategy="use_fallback",
            set_fallback_to_nan=False,
        )
        
        current_atom_count = processed_mol.mol.GetNumAtoms()
        mask_length = len(processed_mol.in_crop_mask)
        
        self.assertEqual(
            mask_length,
            current_atom_count,
            f"in_crop_mask length ({mask_length}) does not match molecule atom count "
            f"({current_atom_count}) with fallback strategy."
        )

    def test_mask_length_after_fallback_to_nan(self):
        """Test that mask length is correct when fallback is set to NaN."""
        mol = self._create_test_molecule_with_hydrogens()
        atom_array = self._create_test_atom_array(mol)
        
        processed_mol = get_processed_reference_conformer(
            mol=mol,
            mol_atom_array=atom_array,
            preferred_confgen_strategy="use_fallback",
            set_fallback_to_nan=True,
        )
        
        current_atom_count = processed_mol.mol.GetNumAtoms()
        mask_length = len(processed_mol.in_crop_mask)
        
        self.assertEqual(
            mask_length,
            current_atom_count,
            f"in_crop_mask length ({mask_length}) does not match molecule atom count "
            f"({current_atom_count}) after setting fallback to NaN."
        )

    def test_mask_can_be_zipped_with_atoms(self):
        """Test that in_crop_mask can be safely zipped with mol.GetAtoms()."""
        mol = self._create_test_molecule_with_hydrogens()
        atom_array = self._create_test_atom_array(mol)
        
        processed_mol = get_processed_reference_conformer(
            mol=mol,
            mol_atom_array=atom_array,
            preferred_confgen_strategy="default",
            set_fallback_to_nan=False,
        )
        
        # This should not raise ValueError
        atoms = list(processed_mol.mol.GetAtoms())
        masks = list(processed_mol.in_crop_mask)
        
        self.assertEqual(
            len(atoms),
            len(masks),
            "Cannot zip atoms and masks - lengths don't match."
        )
        
        # Verify we can zip with strict=True (Python 3.10+)
        try:
            zipped = list(zip(atoms, masks, strict=True))
            self.assertEqual(len(zipped), len(atoms))
        except ValueError as e:
            self.fail(f"Failed to zip atoms and masks with strict=True: {e}")

    def test_mask_length_with_complex_molecule(self):
        """Test that mask length is correct with a complex molecule containing hydrogens."""
        # Complex molecule with explicit hydrogens in SMILES
        complex_smiles = "O=C(CCN1CCN(C(=O)OCc2cc(Cl)cc(Cl)c2)CC1)c1ccc2[nH]c(=O)oc2c1"
        mol = self._create_test_molecule_with_hydrogens(complex_smiles)
        atom_array = self._create_test_atom_array(mol)
        
        # Get initial atom count (with hydrogens)
        initial_atom_count = mol.GetNumAtoms()
        
        # Create processed reference conformer with default strategy (removes Hs)
        processed_mol = get_processed_reference_conformer(
            mol=mol,
            mol_atom_array=atom_array,
            preferred_confgen_strategy="default",
            set_fallback_to_nan=False,
        )
        
        # Check that mask length matches current molecule atom count
        current_atom_count = processed_mol.mol.GetNumAtoms()
        mask_length = len(processed_mol.in_crop_mask)
        
        self.assertEqual(
            mask_length,
            current_atom_count,
            f"in_crop_mask length ({mask_length}) does not match molecule atom count "
            f"({current_atom_count}). Initial count was {initial_atom_count}."
        )
        
        # Verify we can iterate without error
        for atom, mask in zip(processed_mol.mol.GetAtoms(), processed_mol.in_crop_mask, strict=True):
            self.assertIsNotNone(atom)
            self.assertIsInstance(mask, (bool, np.bool_))

    def test_mask_length_with_complex_molecule_fallback(self):
        """Test that mask length is correct with complex molecule using fallback."""
        complex_smiles = "O=C(CCN1CCN(C(=O)OCc2cc(Cl)cc(Cl)c2)CC1)c1ccc2[nH]c(=O)oc2c1"
        mol = self._create_test_molecule_with_hydrogens(complex_smiles)
        atom_array = self._create_test_atom_array(mol)
        
        processed_mol = get_processed_reference_conformer(
            mol=mol,
            mol_atom_array=atom_array,
            preferred_confgen_strategy="use_fallback",
            set_fallback_to_nan=False,
        )
        
        current_atom_count = processed_mol.mol.GetNumAtoms()
        mask_length = len(processed_mol.in_crop_mask)
        
        self.assertEqual(
            mask_length,
            current_atom_count,
            f"in_crop_mask length ({mask_length}) does not match molecule atom count "
            f"({current_atom_count}) for complex molecule with fallback strategy."
        )


if __name__ == "__main__":
    unittest.main()

